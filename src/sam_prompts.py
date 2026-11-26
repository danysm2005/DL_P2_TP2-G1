sam_prompts.py
100%
"""SAM / SAM 2 zero-shot com prompts oráculo e com jitter (A3)."""
import argparse
import time
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from src.avaliar import dice
from src.dados import KVASIR, carregar_divisao, ler_imagem, ler_mascara


def carregar_sam(modelo, ckpt, device="cuda"):
    if modelo == "sam_vit_b":
        from segment_anything import SamPredictor, sam_model_registry
        return SamPredictor(sam_model_registry["vit_b"](checkpoint=ckpt).to(device).eval())
    if modelo == "sam2_hiera_s":
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        # o nome do config depende da versão instalada (sam2 vs sam2.1)
        return SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", ckpt, device=device))
    raise ValueError(modelo)


def caixa_de_mascara(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)


def jitter_caixa(box, jitter, rng, hw):
    x0, y0, x1, y1 = box
    if jitter:
        dx, dy = jitter * (x1 - x0), jitter * (y1 - y0)
        x0, x1 = x0 + rng.uniform(-dx, dx), x1 + rng.uniform(-dx, dx)
        y0, y1 = y0 + rng.uniform(-dy, dy), y1 + rng.uniform(-dy, dy)
    h, w = hw
    return np.clip([x0, y0, x1, y1], 0, [w - 1, h - 1, w - 1, h - 1]).astype(np.float32)


def caixa(mask, jitter=0.0, rng=None):
    return jitter_caixa(caixa_de_mascara(mask), jitter, rng, mask.shape)


def ponto(mask, centro=True, rng=None):
    if centro:  # ponto mais interior do pólipo
        d = distance_transform_edt(mask > 0)
        y, x = np.unravel_index(d.argmax(), d.shape)
    else:
        ys, xs = np.where(mask > 0)
        i = rng.integers(len(ys)); y, x = ys[i], xs[i]
    return np.array([[x, y]], np.float32), np.array([1])


def estabilidade(masks):
    """Dice médio entre as máscaras candidatas (sinal de estabilidade para a app)."""
    return float(np.mean([dice(a, b) for a, b in combinations(masks, 2)]))


def correr(predictor, nomes, out_dir, prompt="caixa", jitter=0.0, n_prompts=1,
           seed=0, root=KVASIR, decoder_afinado=False):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tempos, linhas = [], []
    for i, n in enumerate(nomes):
        img, gt = ler_imagem(n, root), ler_mascara(n, root)
        t0 = time.time()
        predictor.set_image(img)  # o encoder corre uma vez por imagem
        for k in range(n_prompts):
            rng = np.random.default_rng([seed, i, k])  # reprodutível por imagem e prompt
            if prompt == "caixa":
                kw = dict(box=caixa(gt, jitter, rng))
            elif prompt == "ponto_centro":
                kw = dict(zip(("point_coords", "point_labels"), ponto(gt, True)))
            else:
                kw = dict(zip(("point_coords", "point_labels"), ponto(gt, False, rng)))
            masks, scores, _ = predictor.predict(multimask_output=True, **kw)
            linhas.append((n, k, float(np.max(scores))))
            m = masks[int(np.argmax(scores))]  # maior score previsto, NUNCA a mais parecida com a real
            cv2.imwrite(str(out_dir / f"{n}_p{k}.png"), (m > 0).astype(np.uint8) * 255)
        tempos.append(time.time() - t0)
    import pandas as pd
    pd.DataFrame(linhas, columns=["nome", "k", "score"]).to_csv(out_dir / "scores.csv", index=False)
    print(f"{len(nomes)} imagens, {np.mean(tempos):.2f} s/imagem")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="sam_vit_b", choices=["sam_vit_b", "sam2_hiera_s"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="caixa", choices=["caixa", "ponto_centro", "ponto_aleatorio"])
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--part", default="val", choices=["val", "test"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    n_p = a.n_prompts or (5 if (a.jitter > 0 or a.prompt == "ponto_aleatorio") else 1)
    pred = carregar_sam(a.modelo, a.ckpt)
    nomes = carregar_divisao(a.split)[a.part]
    pasta = f"/content/drive/MyDrive/P2_unet/runs_SAMsemfine/{a.modelo}/{a.prompt}_j{a.jitter}/split{a.split}_{a.part}"
    correr(pred, nomes, pasta, a.prompt, a.jitter, n_p, a.seed)
    print("avaliar com: python -m src.avaliar --pred", pasta, "--split", a.split,
          "--part", a.part, "--n-prompts", n_p)
A apresentar sam_prompts.py.
