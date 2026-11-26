"""Fine-tuning leve do decoder: SAM1/SAM2 x caixa/ponto_centro. Só split 1, seed 1 por defeito."""
import argparse, copy, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch, torch.nn.functional as F, yaml

from src.avaliar import dice
from src.dados import carregar_divisao, ler_imagem, ler_mascara
from src.sam_prompts import caixa_de_mascara, carregar_sam, jitter_caixa, ponto

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BASE = Path("/content/drive/MyDrive/P2_unet")
CKPT = {"sam_vit_b": "/content/sam_vit_b_01ec64.pth", "sam2_hiera_s": "/content/sam2.1_hiera_small.pt"}
CFG = dict(epocas=30, paciencia=5, lr=5e-5, wd=0.01, bs=8, jitter_treino=0.1)


def prompt(gt, tipo, rng=None, jitter=0.0):
    if tipo == "caixa":
        return ("caixa", jitter_caixa(caixa_de_mascara(gt), jitter, rng, gt.shape))
    c, l = ponto(gt, True)
    return ("ponto", c, l)


class SAM1:
    def __init__(self, pred):
        from segment_anything.utils.transforms import ResizeLongestSide
        self.pred, self.model, self.decoder = pred, pred.model, pred.model.mask_decoder
        self.tr, self.cache = ResizeLongestSide(self.model.image_encoder.img_size), {}

    def features(self, n):
        if n not in self.cache:  # encoder congelado: 1 vez por imagem (fp16 na RAM)
            img = ler_imagem(n); self.pred.set_image(img)
            self.cache[n] = dict(emb=self.pred.features.half().cpu(), orig=tuple(img.shape[:2]),
                                 inp=tuple(self.pred.input_size))
        return self.cache[n]

    def decode(self, f, pr):
        if pr[0] == "caixa":
            b = torch.as_tensor(self.tr.apply_boxes(pr[1][None].astype(np.float32), f["orig"]), device=DEV)
            kw = dict(points=None, boxes=b, masks=None)
        else:
            c = torch.as_tensor(self.tr.apply_coords(pr[1].astype(np.float32), f["orig"]), device=DEV)[None]
            l = torch.as_tensor(pr[2], dtype=torch.int, device=DEV)[None]
            kw = dict(points=(c, l), boxes=None, masks=None)
        with torch.no_grad():
            sp, de = self.model.prompt_encoder(**kw)
        low, _ = self.decoder(image_embeddings=f["emb"].to(DEV).float(),
                              image_pe=self.model.prompt_encoder.get_dense_pe(),
                              sparse_prompt_embeddings=sp, dense_prompt_embeddings=de,
                              multimask_output=False)
        return low

    def full(self, low, f):
        return self.model.postprocess_masks(low, f["inp"], f["orig"])[0, 0]

    def alvo(self, m):  # referencial do low_res do SAM1: lado maior = 256, padding em baixo/direita
        h, w = m.shape; s = 256 / max(h, w); nh, nw = round(h * s), round(w * s)
        out = np.zeros((256, 256), np.float32)
        out[:nh, :nw] = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
        return out


class SAM2:
    def __init__(self, pred):
        self.pred, self.model, self.decoder = pred, pred.model, pred.model.sam_mask_decoder

    def features(self, n):  # encoder congelado, corre em cada passo (sem cache: usa muita RAM)
        img = ler_imagem(n); self.pred.set_image(img)
        return dict(feats=dict(self.pred._features), orig=tuple(self.pred._orig_hw[-1]))

    def decode(self, f, pr):
        T = self.pred._transforms
        if pr[0] == "caixa":  # o SAM2 trata a caixa como 2 pontos com etiquetas 2 e 3
            b = torch.as_tensor(pr[1][None], dtype=torch.float, device=DEV)
            coords = T.transform_boxes(b, normalize=True, orig_hw=f["orig"]).reshape(-1, 2, 2)
            labels = torch.tensor([[2, 3]], dtype=torch.int, device=DEV)
        else:
            c = torch.as_tensor(pr[1], dtype=torch.float, device=DEV)[None]
            coords = T.transform_coords(c, normalize=True, orig_hw=f["orig"])
            labels = torch.as_tensor(pr[2], dtype=torch.int, device=DEV)[None]
        with torch.no_grad():
            sp, de = self.model.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        hr = [x[-1].unsqueeze(0) for x in f["feats"]["high_res_feats"]]
        low, _, _, _ = self.decoder(image_embeddings=f["feats"]["image_embed"][-1].unsqueeze(0),
                                    image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                                    sparse_prompt_embeddings=sp, dense_prompt_embeddings=de,
                                    multimask_output=False, repeat_image=False, high_res_features=hr)
        return low

    def full(self, low, f):
        return self.pred._transforms.postprocess_masks(low, f["orig"])[0, 0]

    def alvo(self, m):  # o SAM2 redimensiona a imagem para 1024x1024 sem manter proporção
        return cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.float32)


def dice_bce(logits, y):
    p = torch.sigmoid(logits)
    d = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return d + F.binary_cross_entropy_with_logits(logits, y)


@torch.no_grad()
def validar(W, nomes, gts, tipo):
    ds = []
    for n in nomes:
        f = W.features(n)
        m = (W.full(W.decode(f, prompt(gts[n], tipo)), f) > 0).cpu().numpy()
        ds.append(dice(gts[n], m))
    return float(np.mean(ds))


def treinar(W, d, gts, tipo, seed, out):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    for p in W.model.parameters():
        p.requires_grad_(False)
    for p in W.decoder.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(W.decoder.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    v0 = validar(W, d["val"], gts, tipo)
    print(f"val mDice zero-shot (1 máscara) = {v0:.4f}", flush=True)
    melhor, sem, log = -1.0, 0, []
    for ep in range(CFG["epocas"]):
        ordem = [d["train"][i] for i in rng.permutation(len(d["train"]))]
        l = 0.0; opt.zero_grad()
        for j, n in enumerate(ordem, 1):
            f = W.features(n)
            pr = prompt(gts[n], tipo, rng, rng.uniform(0, CFG["jitter_treino"]))
            y = torch.from_numpy(W.alvo(gts[n]))[None, None].to(DEV)
            loss = dice_bce(W.decode(f, pr), y)
            (loss / CFG["bs"]).backward(); l += loss.item()
            if j % CFG["bs"] == 0 or j == len(ordem):
                opt.step(); opt.zero_grad()
        v = validar(W, d["val"], gts, tipo)
        log.append(dict(epoca=ep, loss=l / len(ordem), val_mdice=v)); print(log[-1], flush=True)
        if v > melhor:
            melhor, sem = v, 0
            torch.save(W.decoder.state_dict(), out / "decoder.pt")  # melhor epoch na VALIDAÇÃO
        else:
            sem += 1
            if sem >= CFG["paciencia"]:
                break
    pd.DataFrame(log).to_csv(out / "historico.csv", index=False)
    (out / "info.yaml").write_text(yaml.safe_dump(dict(
        **CFG, seed=seed, hash_divisao=d["hash"], val_mdice_zero_shot_1mascara=v0,
        val_mdice_afinado=melhor, epocas_feitas=len(log))))
    if melhor <= v0:
        print("AVISO: o Dice de validação não subiu face ao zero-shot")


@torch.no_grad()
def prever(W, nomes, tipo, pasta):
    pasta.mkdir(parents=True, exist_ok=True)
    for n in nomes:
        gt, f = ler_mascara(n), W.features(n)
        m = (W.full(W.decode(f, prompt(gt, tipo)), f) > 0).cpu().numpy()
        cv2.imwrite(str(pasta / f"{n}_p0.png"), m.astype(np.uint8) * 255)
    print("previsões em", pasta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["treinar", "prever"])
    ap.add_argument("--modelo", required=True, choices=list(CKPT))
    ap.add_argument("--prompt", required=True, choices=["caixa", "ponto_centro"])
    ap.add_argument("--particao", default="val", choices=["val", "test"])
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--protocolo-fechado", action="store_true")
    a = ap.parse_args()
    out = BASE / "runs_SAMfine" / f"{a.modelo}_{a.prompt}" / f"split{a.split}_seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    W = (SAM1 if a.modelo == "sam_vit_b" else SAM2)(carregar_sam(a.modelo, CKPT[a.modelo], DEV))
    d = carregar_divisao(a.split)
    if a.modo == "treinar":
        gts = {n: ler_mascara(n) for n in d["train"] + d["val"]}
        treinar(W, d, gts, a.prompt, a.seed, out)
    else:
        if a.particao == "test" and not a.protocolo_fechado:
            sys.exit("Test só depois do commit do PROTOCOLO.md: use --protocolo-fechado")
        W.decoder.load_state_dict(torch.load(out / "decoder.pt", map_location=DEV))
        prever(W, d[a.particao], a.prompt, out / f"pred_{a.particao}")
