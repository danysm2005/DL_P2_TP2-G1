"""Afinação leve do descodificador do SAM ViT-B (A4), conforme o PROTOCOLO (secções 4, 6, 8).

- encoder de imagem e encoder de prompts congelados; embeddings pré-calculados uma vez
- perda Dice+BCE; early stopping no mDice de VALIDAÇÃO (caixa oráculo)
- 3 divisões x 3 seeds (configs/sam_ft.yaml); config + hash da divisão guardados por corrida
- verificações do protocolo: sem gradientes no encoder, loss a descer, Dice val vs zero-shot
- 'prever' grava PNGs na resolução original, com os MESMOS prompts (seed) do zero-shot,
  para avaliar com src/avaliar.py (pipeline único)
"""
import argparse
import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from segment_anything.utils.transforms import ResizeLongestSide

from src.avaliar import dice
from src.dados import CVC, KVASIR, carregar_divisao, ler_imagem, ler_mascara, listar
from src.sam_prompts import caixa_de_mascara, carregar_sam, jitter_caixa

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PASTA_EMB = Path("/content/runs/emb/sam_vit_b")
DEFAULTS = dict(modelo="sam_vit_b", ckpt="/content/sam_vit_b_01ec64.pth", prompt="caixa",
                splits=[1, 2, 3], seeds=[1, 2, 3], epocas=30, paciencia=5, lr=5e-5,
                wd=0.01, bs=8, jitter_treino=0.1, seed_prompts=0)


# ---- embeddings -----------------------------------------------------------
def pre_calcular(predictor, nomes, pasta=PASTA_EMB, root=KVASIR):
    """1 corrida do encoder por imagem; (1,256,64,64) em fp16 = ~2 MB por imagem."""
    pasta.mkdir(parents=True, exist_ok=True)
    for n in nomes:
        f = pasta / f"{n}.pt"
        if f.exists():
            continue
        img = ler_imagem(n, root)
        predictor.set_image(img)
        torch.save(dict(emb=predictor.features.half().cpu(), orig=tuple(img.shape[:2]),
                        inp=tuple(predictor.input_size)), f)


def carregar_emb(nomes, pasta=PASTA_EMB):
    return {n: torch.load(pasta / f"{n}.pt") for n in nomes}


# ---- passo do descodificador ------------------------------------------------
def alvo_256(mask):
    """Máscara real no referencial do low_res do SAM (lado maior = 256, padding em baixo/direita)."""
    h, w = mask.shape
    s = 256 / max(h, w)
    nh, nw = round(h * s), round(w * s)
    out = np.zeros((256, 256), np.float32)
    out[:nh, :nw] = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return out


def prever_low(sam, tr, emb, box, orig):
    b = torch.as_tensor(tr.apply_boxes(box[None].astype(np.float32), orig), device=DEV)
    with torch.no_grad():  # encoder de prompts congelado
        sp, de = sam.prompt_encoder(points=None, boxes=b, masks=None)
    low, _ = sam.mask_decoder(image_embeddings=emb.to(DEV).float(),
                              image_pe=sam.prompt_encoder.get_dense_pe(),
                              sparse_prompt_embeddings=sp, dense_prompt_embeddings=de,
                              multimask_output=False)
    return low  # logits (1,1,256,256)


def dice_bce(logits, y):
    p = torch.sigmoid(logits)
    d = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return d + F.binary_cross_entropy_with_logits(logits, y)


@torch.no_grad()
def validar(sam, tr, nomes, embs, gts):
    """mDice com caixa oráculo, resolução original, uma máscara (multimask_output=False)."""
    ds = []
    for n in nomes:
        e = embs[n]
        low = prever_low(sam, tr, e["emb"], caixa_de_mascara(gts[n]), e["orig"])
        m = sam.postprocess_masks(low, e["inp"], e["orig"])[0, 0] > 0
        ds.append(dice(gts[n], m.cpu().numpy()))
    return float(np.mean(ds))


def verificar_congelamento(sam):
    assert all(p.grad is None for p in sam.image_encoder.parameters()), "o encoder recebeu gradientes!"
    assert all(p.grad is None for p in sam.prompt_encoder.parameters()), "o prompt encoder recebeu gradientes!"
    n = sum(p.grad is not None for p in sam.mask_decoder.parameters())
    assert n > 0, "o descodificador não recebeu gradientes"
    print(f"[verificação] OK: encoders sem gradientes; {n} tensores do descodificador com gradiente")


# ---- treino -------------------------------------------------------------------
def afinar(cfg, split, seed, pred, estado0, embs, gts):
    sam = pred.model
    d = carregar_divisao(split)
    sam.mask_decoder.load_state_dict(copy.deepcopy(estado0))  # cada corrida parte dos pesos originais
    for p in sam.parameters():
        p.requires_grad_(False)
    for p in sam.mask_decoder.parameters():
        p.requires_grad_(True)
    tr = ResizeLongestSide(sam.image_encoder.img_size)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    out = Path(f"/content/drive/MyDrive/P2_unet/runs_SAMfine/split{split}_seed{seed}"); out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump({**cfg, "split": split, "seed": seed,
                                                     "hash_divisao": d["hash"]}))
    opt = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    v0 = validar(sam, tr, d["val"], embs, gts)  # zero-shot, MESMO pipeline (uma máscara, caixa oráculo)
    print(f"[split{split} seed{seed}] val mDice zero-shot (1 máscara) = {v0:.4f}")
    melhor, sem, log = -1.0, 0, []
    for ep in range(cfg["epocas"]):
        ordem, l = rng.permutation(d["train"]), 0.0
        opt.zero_grad()
        for j, n in enumerate(ordem, 1):
            e = embs[n]
            box = jitter_caixa(caixa_de_mascara(gts[n]), rng.uniform(0, cfg["jitter_treino"]),
                               rng, e["orig"])
            y = torch.from_numpy(alvo_256(gts[n]))[None, None].to(DEV)
            loss = dice_bce(prever_low(sam, tr, e["emb"], box, e["orig"]), y)
            (loss / cfg["bs"]).backward()
            l += loss.item()
            if ep == 0 and j == 1:
                verificar_congelamento(sam)
            if j % cfg["bs"] == 0 or j == len(ordem):
                opt.step(); opt.zero_grad()
        v = validar(sam, tr, d["val"], embs, gts)
        log.append(dict(epoca=ep, loss=l / len(ordem), val_mdice=v)); print(log[-1])
        if v > melhor:
            melhor, sem = v, 0
            torch.save(sam.mask_decoder.state_dict(), out / "decoder.pt")
        else:
            sem += 1
            if sem >= cfg["paciencia"]:
                break
    if melhor <= v0:
        print("AVISO: o Dice de validação não subiu face ao zero-shot - rever lr/alinhamento do alvo")
    (out / "info.yaml").write_text(yaml.safe_dump(dict(
        hash_divisao=d["hash"], val_mdice_zero_shot=v0, val_mdice_afinado=melhor,
        ganho=melhor - v0, epocas_feitas=len(log), log=log)))


# ---- inferência (PNGs para o avaliar.py) -----------------------------------------
@torch.no_grad()
def prever(cfg, split, seed, particao, jitter, n_prompts, pred):
    sam = pred.model
    pasta = Path(f"/content/drive/MyDrive/P2_unet/runs_SAMfine/split{split}_seed{seed}")
    sam.mask_decoder.load_state_dict(torch.load(pasta / "decoder.pt", map_location=DEV))
    tr = ResizeLongestSide(sam.image_encoder.img_size)
    if particao == "cvc":
        root, nomes = CVC, listar(CVC)
    else:
        root, nomes = KVASIR, carregar_divisao(split)[particao]
    out = pasta / f"pred_{particao}_j{jitter}"; out.mkdir(exist_ok=True)
    for i, n in enumerate(nomes):
        img, gt = ler_imagem(n, root), ler_mascara(n, root)
        pred.set_image(img)
        orig, inp = tuple(img.shape[:2]), pred.input_size
        for k in range(n_prompts):
            rng = np.random.default_rng([cfg["seed_prompts"], i, k])  # = sam_prompts.py
            box = jitter_caixa(caixa_de_mascara(gt), jitter, rng, orig)
            low = prever_low(sam, tr, pred.features, box, orig)
            m = (sam.postprocess_masks(low, inp, orig)[0, 0] > 0).cpu().numpy()
            cv2.imwrite(str(out / f"{n}_p{k}.png"), m.astype(np.uint8) * 255)
    print("previsões em", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["treinar", "prever"])
    ap.add_argument("--config", default="/content/drive/MyDrive/P2_unet/configs/sam_ft.yaml")
    ap.add_argument("--splits", type=int, nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--particao", default="val", choices=["val", "test", "cvc"])
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--n-prompts", type=int)
    ap.add_argument("--protocolo-fechado", action="store_true",
                    help="obrigatório para test/cvc: PROTOCOLO.md já está em commit")
    a = ap.parse_args()
    cfg = {**DEFAULTS, **(yaml.safe_load(open(a.config)) if Path(a.config).exists() else {})}
    splits, seeds = a.splits or cfg["splits"], a.seeds or cfg["seeds"]
    pred = carregar_sam(cfg["modelo"], cfg["ckpt"], DEV)
    if a.modo == "treinar":
        todos = listar()
        pre_calcular(pred, todos)
        embs = carregar_emb(todos)  # ~2 GB de RAM (fp16)
        gts = {n: ler_mascara(n) for n in todos}
        estado0 = copy.deepcopy(pred.model.mask_decoder.state_dict())
        for s in splits:
            for sd in seeds:
                afinar(cfg, s, sd, pred, estado0, embs, gts)
    else:
        if a.particao in ("test", "cvc") and not a.protocolo_fechado:
            sys.exit("Teste/CVC só depois do commit do PROTOCOLO.md: use --protocolo-fechado "
                     "(uma única corrida por condição e divisão).")
        n_p = a.n_prompts or (5 if a.jitter > 0 else 1)
        for s in splits:
            for sd in seeds:
                prever(cfg, s, sd, a.particao, a.jitter, n_p, pred)
