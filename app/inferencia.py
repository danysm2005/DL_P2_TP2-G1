"""Inferência para a interface: U-Net, SAM e SAM 2, cada um com e sem afinação.

Contrato:  predict(img, prompt, modelo) -> (masks[3,H,W] bool, scores[3])
  img     : np.ndarray HxWx3 uint8 RGB, resolução original
  prompt  : {"ponto": (x, y)} | {"caixa": (x0, y0, x1, y1)} | {"auto": True}
  modelo  : um dos valores de MODELOS
"""
import os
import re

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from pathlib import Path
import numpy as np

BASE = Path(os.environ.get("P2_BASE", "/content/drive/MyDrive/P2_unet"))
LOCAIS_BASE = [BASE, Path("/content")]
DIR_AFINADOS = BASE / "runs_SAMfine"
IMG_UNET = (352, 352)

MODELOS = ["U-Net", "SAM", "SAM afinado", "SAM 2", "SAM 2 afinado"]
_CHAVES = {
    "U-Net": "unet",
    "SAM": "sam",
    "SAM afinado": "sam_ft",
    "SAM 2": "sam2",
    "SAM 2 afinado": "sam2_ft",
}

_cache, _imagem_atual, _simulados, _detalhes, _estado = {}, {}, set(), {}, {}


def modelos_simulados():
    return set(_simulados)


def detalhe_modelo(modelo):
    chave = _CHAVES.get(modelo, "")
    return _detalhes.get(chave, "")


# --------------------------------------------------------------------------- #
# Procura de checkpoints
# --------------------------------------------------------------------------- #
def _escolher_seed_unet():
    """Seed com melhor mDice de VALIDAÇÃO (nunca o teste). UNET_SEED=2 força uma."""
    if os.environ.get("UNET_SEED"):
        return int(os.environ["UNET_SEED"])
    melhor, seed = -1.0, 1
    for s in (1, 2, 3):
        npz, ck = BASE / f"unet_val_seed{s}.npz", BASE / f"unet_seed{s}_final.keras"
        if npz.exists() and ck.exists():
            d = float(np.mean(np.load(npz)["dice"]))
            print(f"[inferencia] U-Net seed {s}: mDice validação = {d:.4f}")
            if d > melhor:
                melhor, seed = d, s
    return seed


def _achar_sam_base():
    if os.environ.get("SAM_CKPT"):
        return Path(os.environ["SAM_CKPT"])
    for pasta in LOCAIS_BASE:
        p = pasta / "sam_vit_b_01ec64.pth"
        if p.exists():
            return p
    return BASE / "sam_vit_b_01ec64.pth"


def _achar_sam2_base():
    if os.environ.get("SAM2_CKPT"):
        return Path(os.environ["SAM2_CKPT"])
    for pasta in LOCAIS_BASE:
        cands = sorted(pasta.glob("sam2*.pt"))
        if cands:
            return cands[0]
    return None


def _checkpoints_afinados():
    """Todos os *.pt/*.pth até 3 níveis dentro de runs_SAMfine."""
    if not DIR_AFINADOS.exists():
        return []
    achados = set()
    for padrao in ("*", "*/*", "*/*/*"):
        for ext in (".pt", ".pth"):
            achados.update(DIR_AFINADOS.glob(padrao + ext))
    return sorted(achados)


def _achar_afinado(sam2, tipo):
    """Encontra o checkpoint afinado específico para (sam1/sam2) e (caixa/ponto)."""
    env_var = f"{'SAM2' if sam2 else 'SAM'}_AFINADO_{tipo.upper()}"
    if os.environ.get(env_var):
        return Path(os.environ[env_var])
    env_geral = "SAM2_AFINADO" if sam2 else "SAM_AFINADO"
    if os.environ.get(env_geral):
        return Path(os.environ[env_geral])

    cands = [p for p in _checkpoints_afinados() if ("sam2" in str(p).lower()) == sam2]
    if tipo == "caixa":
        por_tipo = [p for p in cands if "caixa" in str(p).lower() or "box" in str(p).lower()]
    else:
        por_tipo = [p for p in cands if "ponto" in str(p).lower() or "point" in str(p).lower()]

    cands = por_tipo or cands
    pref = [p for p in cands if "split1_seed1" in str(p)] or cands
    return pref[0] if pref else None


# --------------------------------------------------------------------------- #
# Carregamento (uma só vez)
# --------------------------------------------------------------------------- #
def _carregar_unet():
    import tensorflow as tf
    try:
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError:
        pass
    try:
        import tf_keras as keras
    except ImportError:
        from tensorflow import keras

    s = _escolher_seed_unet()
    model = keras.models.load_model(BASE / f"unet_seed{s}_final.keras", compile=False)
    npz = BASE / f"unet_val_seed{s}.npz"
    thr = float(np.load(npz)["thr"]) if npz.exists() else 0.5
    _estado["seed_unet"] = s
    _detalhes["unet"] = f"seed {s} · thr {thr:.2f}"
    print(f"[inferencia] U-Net carregada (tf_keras): seed {s}, threshold {thr}")
    return model, thr


def _aplicar_pesos(rede, caminho, decoder_attr):
    import torch
    ck = torch.load(caminho, map_location="cpu")
    sd = ck.get("state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck.state_dict()
    alvo = rede if any(k.startswith(decoder_attr + ".") for k in sd) else getattr(rede, decoder_attr)
    r = alvo.load_state_dict(sd, strict=False)
    print(f"[inferencia]   pesos de {Path(caminho).name}: em falta={len(r.missing_keys)}, inesperadas={len(r.unexpected_keys)}")


def _carregar_sam(afinado, tipo=None):
    import torch
    from segment_anything import SamPredictor, sam_model_registry
    ckpt_sam = _achar_sam_base()
    if not ckpt_sam.exists():
        raise FileNotFoundError(f"sem sam_vit_b_01ec64.pth em {LOCAIS_BASE}")
    sam = sam_model_registry["vit_b"](checkpoint=str(ckpt_sam))
    if afinado:
        ck = _achar_afinado(sam2=False, tipo=tipo)
        if ck is None:
            raise FileNotFoundError(f"sem checkpoint SAM afinado ({tipo}) em {DIR_AFINADOS}")
        _aplicar_pesos(sam, ck, "mask_decoder")
        _detalhes[f"sam_ft_{tipo}"] = f"ViT-B · FT {tipo} ({ck.parent.name})"
        _detalhes["sam_ft"] = f"ViT-B · FT ({tipo})"
    else:
        _detalhes["sam"] = "ViT-B · Zero-Shot"
    sam.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    print(f"[inferencia] {'SAM afinado (' + tipo + ')' if afinado else 'SAM'} carregado")
    return SamPredictor(sam)


def _config_sam2(ck):
    nome = ck.name.lower()
    tam = "t" if "tiny" in nome else "b+" if ("base" in nome or "b+" in nome) else "l" if "large" in nome else "s"
    if os.environ.get("SAM2_CFG"):
        return os.environ["SAM2_CFG"]
    return f"configs/sam2.1/sam2.1_hiera_{tam}.yaml" if "2.1" in nome else f"configs/sam2/sam2_hiera_{tam}.yaml"


def _carregar_sam2(afinado, tipo=None):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    base = _achar_sam2_base()
    if base is None:
        raise FileNotFoundError(f"sem sam2*.pt em {LOCAIS_BASE}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rede = build_sam2(_config_sam2(base), str(base), device=dev)
    if afinado:
        ck = _achar_afinado(sam2=True, tipo=tipo)
        if ck is None:
            raise FileNotFoundError(f"sem checkpoint SAM 2 afinado ({tipo}) em {DIR_AFINADOS}")
        _aplicar_pesos(rede, ck, "sam_mask_decoder")
        _detalhes[f"sam2_ft_{tipo}"] = f"Hiera · FT {tipo} ({ck.parent.name})"
        _detalhes["sam2_ft"] = f"Hiera · FT ({tipo})"
    else:
        _detalhes["sam2"] = f"Hiera · Zero-Shot ({base.name})"
    print(f"[inferencia] {'SAM 2 afinado (' + tipo + ')' if afinado else 'SAM 2'} carregado")
    return SAM2ImagePredictor(rede)


_CARREGADORES = {
    "unet": _carregar_unet,
    "sam": lambda: _carregar_sam(False),
    "sam_ft_caixa": lambda: _carregar_sam(True, "caixa"),
    "sam_ft_ponto": lambda: _carregar_sam(True, "ponto"),
    "sam2": lambda: _carregar_sam2(False),
    "sam2_ft_caixa": lambda: _carregar_sam2(True, "caixa"),
    "sam2_ft_ponto": lambda: _carregar_sam2(True, "ponto"),
}


def _get(chave):
    if chave not in _cache:
        try:
            _cache[chave] = _CARREGADORES[chave]()
        except Exception as e:
            print(f"[inferencia] '{chave}' indisponível ({type(e).__name__}: {e}) -> STUB")
            _cache[chave] = None
    return _cache[chave]


def pre_carregar():
    print("[inferencia] checkpoints afinados encontrados:", [str(p.relative_to(BASE)) for p in _checkpoints_afinados()] or "nenhum")
    for k in _CARREGADORES:
        _get(k)
    _ler_ranking(diagnostico=True)


# --------------------------------------------------------------------------- #
# Inferência
# --------------------------------------------------------------------------- #
def _unet(img):
    import tensorflow as tf
    model, thr = _get("unet")
    h, w = img.shape[:2]
    x = tf.image.resize(img, IMG_UNET) / 255.0
    p = model.predict(x[None], verbose=0)[0]
    p = tf.image.resize(p, (h, w)).numpy()[..., 0]
    return p > thr, p


def _sam(img, prompt, chave, multimask=True):
    import torch
    pred = _get(chave)
    k = (chave, img.shape, hash(img.tobytes()))
    with torch.inference_mode():
        if _imagem_atual.get(chave) != k:
            pred.set_image(img)
            _imagem_atual[chave] = k
        if "caixa" in prompt:
            kw = dict(box=np.array(prompt["caixa"], dtype=np.float32))
        else:
            x, y = prompt["ponto"]
            kw = dict(point_coords=np.array([[x, y]], dtype=np.float32), point_labels=np.array([1]))
        masks, scores, _ = pred.predict(multimask_output=multimask, **kw)

    if not multimask:
        m = np.asarray(masks[0], dtype=bool)
        s = float(scores[0])
        return np.stack([m] * 3), np.array([s] * 3)
    return np.asarray(masks).astype(bool), np.asarray(scores, dtype=float)


def predict(img, prompt, modelo):
    img = np.ascontiguousarray(img[..., :3])
    chave_base = _CHAVES[modelo]
    afinado = modelo.endswith("afinado")
    tp = "caixa" if "caixa" in prompt else "ponto"
    chave = f"{chave_base}_{tp}" if afinado else chave_base

    if _get(chave) is None:
        _simulados.add(modelo)
        return _stub(img, prompt)
    _simulados.discard(modelo)

    if afinado and chave in _detalhes:
        _detalhes[chave_base] = _detalhes[chave]

    if modelo == "U-Net":
        m, p = _unet(img)
        s = float(p[m].mean()) if m.any() else 0.0
        return np.stack([m] * 3), np.array([s] * 3)

    return _sam(img, prompt, chave, multimask=not afinado)


# --------------------------------------------------------------------------- #
# mDice / mIoU de cada modelo no conjunto de TESTE
# --------------------------------------------------------------------------- #
def _ler_ranking(diagnostico=False):
    if "df" not in _estado:
        cands = [
            BASE / "resultados_finais_relatorio" / "tabela_global_3seeds.csv",
            BASE / "ranking_modelos.csv",
        ]
        df = None
        for p in cands:
            if p.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(p, sep=None, engine="python")
                    print(f"[inferencia] tabela de métricas lida: {p}")
                    break
                except Exception as e:
                    print(f"[inferencia] erro ao ler {p} ({type(e).__name__}: {e})")
        _estado["df"] = df
    if diagnostico and _estado["df"] is not None:
        print("[inferencia] colunas:", list(_estado["df"].columns))
        print(_estado["df"].head(15).to_string())
    return _estado["df"]


def _num(serie):
    """Extrai o primeiro número (ex.: '0.9372 ± 0.0038' -> 0.9372)."""
    import pandas as pd
    extraido = serie.astype(str).str.extract(r"([-+]?\d+[.,]?\d*)")[0]
    return pd.to_numeric(extraido.str.replace(",", ".", regex=False), errors="coerce")


def _coluna(df, chave):
    cols = [c for c in df.columns if chave in str(c).lower()
            and not re.search(r"std|desvio|ic\s*95|ci\s*95|bootstrap|falhas|low|high|_lo|_hi", str(c).lower())]
    pref = [c for c in cols if "test" in str(c).lower()] or cols
    return pref[0] if pref else None


def _linha_casa(txt, modelo, prompt):
    sam2 = bool(re.search(r"sam\s*[-_ ]?2", txt))
    unet = "unet" in txt or "u-net" in txt
    sem_ft = bool(re.search(r"sem[_\- ]?(fine|ft|afin)|zero|baseline|\bzs\b", txt))
    ft = (not sem_ft) and bool(re.search(r"fine|\bft\b|_ft|ft\b|afin", txt))
    caixa, ponto = bool(re.search(r"caixa|box", txt)), bool(re.search(r"ponto|point", txt))

    if modelo == "U-Net":
        return unet
    if unet or "sam" not in txt:
        return False
    if sam2 != modelo.startswith("SAM 2") or ft != modelo.endswith("afinado"):
        return False
    return (ponto and not caixa) if prompt == "ponto" else (caixa or not ponto)


def metricas_modelo(modelo, prompt=None):
    try:
        df = _ler_ranking()
        if df is not None:
            tp = "ponto" if (prompt and "ponto" in prompt) else "caixa"
            txt = df.apply(lambda r: " ".join(str(v).lower() for v in r.values if isinstance(v, str)), axis=1)
            sel = df[[_linha_casa(t, modelo, tp) for t in txt]]
            cd, ci = _coluna(df, "dice"), _coluna(df, "iou")
            if not sel.empty and cd is not None and ci is not None:
                return float(_num(sel[cd]).mean()), float(_num(sel[ci]).mean())

        if modelo == "U-Net":
            s = _estado.get("seed_unet") or _escolher_seed_unet()
            f = BASE / f"unet_test_seed{s}.npz"
            if f.exists():
                z = np.load(f)
                if "dice" in z.files and "iou" in z.files:
                    return float(np.mean(z["dice"])), float(np.mean(z["iou"]))
        return None, None
    except Exception as e:
        print(f"[inferencia] métricas de '{modelo}' não calculadas ({type(e).__name__}: {e})")
        return None, None


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
