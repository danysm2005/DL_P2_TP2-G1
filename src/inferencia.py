"""Inferência real para a interface: U-Net (Keras/TF), SAM ViT-B e SAM afinado (PyTorch).

Contrato (igual ao do stub):  predict(img, prompt, modelo) -> (masks[3,H,W] bool, scores[3])
  img     : np.ndarray HxWx3 uint8 RGB, resolução original
  prompt  : {"ponto": (x, y)} | {"caixa": (x0, y0, x1, y1)} | {"auto": True}   (píxeis da imagem original)
  modelo  : "U-Net" | "SAM" | "SAM afinado" | "Automático (U-Net → caixa → SAM)"

Se um checkpoint não existir / falhar a carregar, esse modelo cai para um STUB e fica
marcado em modelos_simulados(): a interface mostra "simulado" em vez de fingir que é real.
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # tem de ser ANTES de importar o tensorflow (a U-Net foi treinada assim)

from pathlib import Path
import numpy as np

# ---- caminhos (ajustar por variável de ambiente ou aqui) ------------------- #
BASE = Path(os.environ.get("P2_BASE", "/content/drive/MyDrive/P2_unet"))
CKPT_UNET = BASE / "unet_seed1_final.keras"
NPZ_UNET = BASE / "unet_val_seed1.npz"                       # contém o threshold escolhido na validação
CKPT_SAM = Path(os.environ.get("SAM_CKPT", "/content/sam_vit_b_01ec64.pth"))   # copiar para o Drive!
DIR_SAM_AFINADO = BASE / "runs_SAMfine" / "split1_seed1"     # ← confirmar com o A4 o ficheiro do checkpoint
IMG_UNET = (352, 352)

_cache, _imagem_atual, _simulados = {}, {}, set()


def modelos_simulados():
    return set(_simulados)


# --------------------------------------------------------------------------- #
# Carregamento (uma só vez)
# --------------------------------------------------------------------------- #
def _carregar_unet():
    import tensorflow as tf
    try:
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)   # não roubar toda a GPU ao PyTorch
    except RuntimeError:
        pass
    from tensorflow import keras
    model = keras.models.load_model(CKPT_UNET, compile=False)
    thr = float(np.load(NPZ_UNET)["thr"]) if NPZ_UNET.exists() else 0.5
    print(f"[inferencia] U-Net carregada (threshold={thr})")
    return model, thr


def _carregar_sam(afinado=False):
    import torch
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=str(CKPT_SAM))
    if afinado:
        cands = sorted(list(DIR_SAM_AFINADO.glob("*.pth")) + list(DIR_SAM_AFINADO.glob("*.pt")))
        if not cands:
            raise FileNotFoundError(f"Sem checkpoint (*.pt/*.pth) em {DIR_SAM_AFINADO}")
        ck = torch.load(cands[-1], map_location="cpu")
        sd = ck.get("state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck.state_dict()
        alvo = sam if any(k.startswith("mask_decoder.") for k in sd) else sam.mask_decoder
        r = alvo.load_state_dict(sd, strict=False)
        print(f"[inferencia] SAM afinado: {cands[-1].name} | em falta={len(r.missing_keys)} inesperadas={len(r.unexpected_keys)}")
    sam.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    print(f"[inferencia] {'SAM afinado' if afinado else 'SAM'} carregado")
    return SamPredictor(sam)


def _get(chave):
    """Devolve o componente carregado, ou None (e regista o motivo) se falhar."""
    if chave not in _cache:
        try:
            _cache[chave] = {"unet": _carregar_unet,
                             "sam": lambda: _carregar_sam(False),
                             "afinado": lambda: _carregar_sam(True)}[chave]()
        except Exception as e:
            print(f"[inferencia] '{chave}' indisponível ({type(e).__name__}: {e}) -> STUB")
            _cache[chave] = None
    return _cache[chave]


def pre_carregar():
    """Carrega tudo no arranque, para o tempo mostrado na interface não incluir o carregamento."""
    for k in ("unet", "sam", "afinado"):
        _get(k)


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
def caixa_de_mascara(mask):
    ys, xs = np.where(mask)
    return None if len(xs) == 0 else (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _unet(img):
    import tensorflow as tf
    model, thr = _get("unet")
    h, w = img.shape[:2]
    x = tf.image.resize(img, IMG_UNET) / 255.0            # igual ao treino (carregar())
    p = model.predict(x[None], verbose=0)[0]
    p = tf.image.resize(p, (h, w)).numpy()[..., 0]        # volta à resolução original
    return p > thr, p


def _sam(img, prompt, chave):
    pred = _get(chave)
    k = (chave, img.shape, hash(img.tobytes()))
    if _imagem_atual.get(chave) != k:                      # encoder só corre 1x por imagem
        pred.set_image(img)
        _imagem_atual[chave] = k
    if "caixa" in prompt:
        kw = dict(box=np.array(prompt["caixa"], dtype=np.float32))
    else:
        x, y = prompt["ponto"]
        kw = dict(point_coords=np.array([[x, y]], dtype=np.float32), point_labels=np.array([1]))
    masks, scores, _ = pred.predict(multimask_output=True, **kw)
    return masks.astype(bool), scores.astype(float)


def _predict_real(img, prompt, modelo):
    if modelo == "U-Net":
        m, p = _unet(img)
        s = float(p[m].mean()) if m.any() else 0.0
        return np.stack([m] * 3), np.array([s] * 3)
    if modelo == "SAM":
        return _sam(img, prompt, "sam")
    if modelo == "SAM afinado":
        return _sam(img, prompt, "afinado")
    # automático: U-Net -> caixa envolvente -> SAM (zero-shot)
    m, _ = _unet(img)
    caixa = caixa_de_mascara(m)
    if caixa is None:                                      # U-Net não viu pólipo: máscara vazia (a interface avisa)
        return np.zeros((3,) + m.shape, bool), np.zeros(3)
    return _sam(img, {"caixa": caixa}, "sam")


_NECESSARIOS = {"U-Net": ["unet"], "SAM": ["sam"], "SAM afinado": ["afinado"]}


def predict(img, prompt, modelo):
    img = np.ascontiguousarray(img[..., :3])
    chaves = _NECESSARIOS.get(modelo, ["unet", "sam"])
    if all(_get(c) is not None for c in chaves):
        _simulados.discard(modelo)
        return _predict_real(img, prompt, modelo)
    _simulados.add(modelo)
    return _stub(img, prompt)


# --------------------------------------------------------------------------- #
# STUB (só quando falta o checkpoint)
# --------------------------------------------------------------------------- #
def _stub(img, prompt):
    h, w = img.shape[:2]
    rng = np.random.default_rng(0)
    if "caixa" in prompt:
        x0, y0, x1, y1 = prompt["caixa"]
        cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2, abs(y1 - y0) / 2
    elif "ponto" in prompt:
        (cx, cy), rx, ry = prompt["ponto"], w * 0.08, h * 0.08
    else:
        cx, cy, rx, ry = w / 2, h / 2, w * 0.12, h * 0.12
    yy, xx = np.ogrid[:h, :w]
    masks, scores = [], []
    for _ in range(3):
        f = 1 + rng.uniform(-0.15, 0.15)
        masks.append(((xx - cx) / max(rx * f, 1)) ** 2 + ((yy - cy) / max(ry * f, 1)) ** 2 <= 1)
        scores.append(float(rng.uniform(0.80, 0.97)))
    return np.stack(masks), np.array(scores)
