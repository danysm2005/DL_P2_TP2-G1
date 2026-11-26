"""Avaliação única (A5): todas as vias passam por aqui, a partir de PNGs em runs/."""
import argparse
import csv
import subprocess
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.dados import CVC, KVASIR, carregar_divisao, ler_imagem, ler_mascara, listar


def dice(y, p, eps=1e-7):
    y, p = y > 0, p > 0
    return (2 * (y & p).sum() + eps) / (y.sum() + p.sum() + eps)


def iou(y, p, eps=1e-7):
    y, p = y > 0, p > 0
    return ((y & p).sum() + eps) / ((y | p).sum() + eps)


def ler_previsao(pred_dir, nome, k=None):
    f = Path(pred_dir) / (f"{nome}.png" if k is None else f"{nome}_p{k}.png")
    return cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)


def metricas_por_imagem(nomes, pred_dir, root=KVASIR, k=None):
    """Uma linha por imagem; previsão e máscara real na resolução original."""
    linhas = []
    for n in nomes:
        gt, p = ler_mascara(n, root), ler_previsao(pred_dir, n, k)
        assert p is not None and p.shape == gt.shape, f"resolução/ficheiro errado: {n}"
        a_real, a_prev = (gt > 0).sum(), (p > 0).sum()
        linhas.append(dict(nome=n, dice=dice(gt, p), iou=iou(gt, p),
                           erro_area=abs(a_prev - a_real) / a_real))
    return pd.DataFrame(linhas)


def bootstrap_ic(d, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), *np.percentile(boot, [2.5, 97.5])


def bootstrap_diff(d_a, d_b, n=1000, seed=0):
    """Diferença emparelhada a-b nas MESMAS imagens (mesmos índices)."""
    rng = np.random.default_rng(seed)
    dif = np.asarray(d_a) - np.asarray(d_b)
    boot = [dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(n)]
    return dif.mean(), *np.percentile(boot, [2.5, 97.5])


# ---- subgrupos -----------------------------------------------------------
def brilho_especular(img, lim=235, frac=0.005):
    """Limiares provisórios: fixar no PROTOCOLO.md depois de ver 20 imagens."""
    return (img.min(axis=2) >= lim).mean() > frac


def nitidez_bordo(img, gt):
    """Gradiente médio (Sobel) ao longo do contorno; valor baixo = bordo pouco nítido."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    k = np.ones((5, 5), np.uint8)
    contorno = (cv2.dilate(gt, k) - cv2.erode(gt, k)) > 0
    return float(mag[contorno].mean())


def caracterizar(nomes, root=KVASIR):
    linhas = []
    for n in nomes:
        img, gt = ler_imagem(n, root), ler_mascara(n, root)
        a = gt.mean()
        linhas.append(dict(nome=n, area_rel=a, pequeno=a < 0.02,
                           especular=brilho_especular(img), nitidez=nitidez_bordo(img, gt)))
    return pd.DataFrame(linhas)


# ---- registo em paper/resultados.csv --------------------------------------
def registar(modelo, prompt, config, seed, divisao, particao, metrica, valor,
             csv_path="paper/resultados.csv"):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = "n/a"
    f = Path(csv_path)
    f.parent.mkdir(parents=True, exist_ok=True)
    novo = not f.exists()
    with f.open("a", newline="") as fh:
        w = csv.writer(fh)
        if novo:
            w.writerow(["modelo", "prompt", "config", "seed", "divisao", "particao",
                        "metrica", "valor", "data", "commit"])
        w.writerow([modelo, prompt, config, seed, divisao, particao, metrica,
                    f"{valor:.6f}", date.today().isoformat(), commit])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="pasta com os PNG previstos")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--part", default="val", choices=["val", "test", "cvc"])
    ap.add_argument("--n-prompts", type=int, default=0, help="0 = sem jitter (ficheiro {nome}.png)")
    ap.add_argument("--modelo", help="se dado, regista as métricas em paper/resultados.csv")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = CVC if args.part == "cvc" else KVASIR
    nomes = listar(CVC) if args.part == "cvc" else carregar_divisao(args.split)[args.part]
    ks = [None] if args.n_prompts == 0 else range(args.n_prompts)
    dfs = [metricas_por_imagem(nomes, args.pred, root, k=k) for k in ks]
    d = np.mean([x.dice.values for x in dfs], axis=0)
    i = np.mean([x.iou.values for x in dfs], axis=0)
    e = np.mean([x.erro_area.values for x in dfs], axis=0)
    md, lo, hi = bootstrap_ic(d)
    mi = bootstrap_ic(i)[0]
    print("mDice %.4f [%.4f, %.4f]" % (md, lo, hi))
    print("mIoU  %.4f" % mi)
    print("erro relativo da área: %.4f" % e.mean())
    print("Dice<0.5: %.1f%% das imagens" % (100 * (d < 0.5).mean()))
    if args.n_prompts:
        print("desvio médio do Dice entre prompts: %.4f" %
              np.std([x.dice.values for x in dfs], axis=0).mean())
    if args.modelo:
        for met, v in [("mDice", md), ("mDice_ic_lo", lo), ("mDice_ic_hi", hi),
                       ("mIoU", mi), ("erro_area", e.mean())]:
            registar(args.modelo, args.prompt, args.config, args.seed, args.split,
                     args.part, met, v)
