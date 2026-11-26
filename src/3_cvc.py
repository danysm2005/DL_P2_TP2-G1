# -*- coding: utf-8 -*-
"""
Avaliação externa exclusiva no CVC-ClinicDB (612 imagens) para os 9 modelos:
- U-Net (3 seeds, com threshold otimizado na validação do Kvasir-SEG)
- SAM 1 e SAM 2 Zero-Shot (Caixa e Ponto)
- SAM 1 e SAM 2 Fine-Tuning (Caixa e Ponto, 3 seeds) — carregamento estrito do decoder

Se estiveres numa sessão limpa do Colab (com GPU T4), corre antes numa célula separada:
!pip install -q albumentations segment-anything hydra-core iopath tf-keras
!pip install -q --no-deps git+https://github.com/facebookresearch/sam2.git
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # necessário antes de importar tf_keras (Keras 2)

import sys
import shutil
import zipfile
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch

# =============================================================================
# 1. SETUP: DRIVE, CVC-CLINICDB E CHECKPOINTS SAM 1 / SAM 2
# =============================================================================
if "google.colab" in sys.modules and not Path("/content/drive/MyDrive").exists():
    from google.colab import drive
    drive.mount("/content/drive")

BASE = Path("/content/drive/MyDrive/P2_unet")
OUT = BASE / "resultados_finais_relatorio"
OUT.mkdir(parents=True, exist_ok=True)

os.chdir(BASE)
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# Salvaguarda: src/dados.py verifica /content/kvasir_dataset ao importar
DEST_KVASIR = Path("/content/kvasir_dataset")
if not list(DEST_KVASIR.rglob("images")):
    zips = (
        list(Path("/content/drive/MyDrive").glob("*archive*.zip"))
        + list(BASE.rglob("*archive*.zip"))
    )
    if zips:
        DEST_KVASIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zips[0], "r") as zf:
            zf.extractall(DEST_KVASIR)

# Copiar CVC-ClinicDB do Drive para disco local (/content/cvc-clinicdb)
CVC_DRIVE = BASE / "data" / "cvc-clinicdb"
CVC_FAST = Path("/content/cvc-clinicdb")
if not (CVC_FAST / "images").exists() or len(list((CVC_FAST / "images").glob("*"))) < 600:
    print("📂 A copiar CVC-ClinicDB do Drive para /content/cvc-clinicdb...")
    shutil.copytree(CVC_DRIVE, CVC_FAST, dirs_exist_ok=True)
print(f"✅ CVC-ClinicDB pronto ({len(list((CVC_FAST / 'images').glob('*')))} imagens)")


# Garantir checkpoints SAM 1 e SAM 2
def obter_ckpt(nome: str, url: str) -> str:
    for p in (Path("/content") / nome, BASE / nome):
        if p.exists():
            return str(p)
    dest = Path("/content") / nome
    print(f"⬇️ A transferir {nome}...")
    subprocess.run(["curl", "-L", "-s", url, "-o", str(dest)], check=True)
    return str(dest)


CKPT = {
    "sam_vit_b": obter_ckpt(
        "sam_vit_b_01ec64.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    ),
    "sam2_hiera_s": obter_ckpt(
        "sam2.1_hiera_small.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    ),
}

import src.dados as dados_mod
dados_mod.CVC = CVC_FAST
from src.dados import ler_imagem, ler_mascara, listar
from src.avaliar import dice, iou
from src.sam_prompts import carregar_sam, caixa, ponto

# Controlo de execução (para o repositório final ficam todas a False;
# se quiseres forçar o recálculo de algum modelo, muda a flag respetiva para True)
FORCAR_UNET = False
FORCAR_SAM2_FT = False
FORCAR_TUDO = False

nomes_cvc = listar(CVC_FAST)


def carregar_decoder_estrito(predictor, run_dir: Path):
    """Carrega o decoder.pt de forma estrita (mask_decoder no SAM1, sam_mask_decoder no SAM2)."""
    ckpt = run_dir / "decoder.pt"
    assert ckpt.exists(), f"Checkpoint FT não encontrado em {ckpt}"
    sd = torch.load(ckpt, map_location="cuda")
    dec = (
        predictor.model.sam_mask_decoder
        if hasattr(predictor.model, "sam_mask_decoder")
        else predictor.model.mask_decoder
    )
    dec.load_state_dict(sd)  # estrito: garante que os pesos afinados entram mesmo


# =============================================================================
# 2. INFERÊNCIA NO CVC-CLINICDB — SAM 1 E SAM 2 (ZERO-SHOT + FINE-TUNING)
# =============================================================================
for mod_cod in ["sam_vit_b", "sam2_hiera_s"]:
    # 2A. Zero-Shot (Caixa + Ponto)
    dir_zs_c = BASE / f"runs_SAMsemfine/{mod_cod}/caixa_j0.0/cvc"
    dir_zs_p = BASE / f"runs_SAMsemfine/{mod_cod}/ponto_centro_j0.0/cvc"
    precisa_zs = (
        FORCAR_TUDO
        or not (dir_zs_c / f"{nomes_cvc[-1]}.png").exists()
        or not (dir_zs_p / f"{nomes_cvc[-1]}.png").exists()
    )
    if precisa_zs:
        print(f"▶ A correr {mod_cod} Zero-Shot (Caixa + Ponto) no CVC-ClinicDB...", flush=True)
        dir_zs_c.mkdir(parents=True, exist_ok=True)
        dir_zs_p.mkdir(parents=True, exist_ok=True)
        pred_zs = carregar_sam(mod_cod, CKPT[mod_cod], device="cuda")
        for n in nomes_cvc:
            img, gt = ler_imagem(n, CVC_FAST), ler_mascara(n, CVC_FAST)
            if (gt > 0).sum() == 0:
                continue
            pred_zs.set_image(img)
            masks_c, scores_c, _ = pred_zs.predict(multimask_output=True, box=caixa(gt, 0.0, None))
            cv2.imwrite(
                str(dir_zs_c / f"{n}.png"),
                (masks_c[int(np.argmax(scores_c))] > 0).astype(np.uint8) * 255,
            )

            pt_c, pt_l = ponto(gt, True)
            masks_p, scores_p, _ = pred_zs.predict(
                multimask_output=True, point_coords=pt_c, point_labels=pt_l
            )
            cv2.imwrite(
                str(dir_zs_p / f"{n}.png"),
                (masks_p[int(np.argmax(scores_p))] > 0).astype(np.uint8) * 255,
            )
        del pred_zs
        torch.cuda.empty_cache()
    else:
        print(f"✓ {mod_cod} Zero-Shot CVC já calculado.")

    # 2B. Fine-Tuning (Caixa + Ponto, 3 seeds)
    pred_ft = None
    for prm in ["caixa", "ponto_centro"]:
        for s in (1, 2, 3):
            run_ft = BASE / f"runs_SAMfine/{mod_cod}_{prm}/split{s}_seed{s}"
            dir_ft_cvc = run_ft / "pred_cvc"
            forcar_este = FORCAR_TUDO or (FORCAR_SAM2_FT and mod_cod == "sam2_hiera_s")
            if not forcar_este and (dir_ft_cvc / f"{nomes_cvc[-1]}.png").exists():
                print(f"✓ {mod_cod} FT ({prm}, Seed {s}) CVC já calculado.")
                continue

            print(f"▶ A correr {mod_cod} FT ({prm}, Seed {s}) no CVC-ClinicDB...", flush=True)
            if forcar_este and dir_ft_cvc.exists():
                shutil.rmtree(dir_ft_cvc)
            dir_ft_cvc.mkdir(parents=True, exist_ok=True)
            if pred_ft is None:
                pred_ft = carregar_sam(mod_cod, CKPT[mod_cod], device="cuda")
            carregar_decoder_estrito(pred_ft, run_ft)

            for n in nomes_cvc:
                img, gt = ler_imagem(n, CVC_FAST), ler_mascara(n, CVC_FAST)
                if (gt > 0).sum() == 0:
                    continue
                pred_ft.set_image(img)
                if prm == "caixa":
                    masks, _, _ = pred_ft.predict(multimask_output=False, box=caixa(gt, 0.0, None))
                else:
                    pt_c, pt_l = ponto(gt, True)
                    masks, _, _ = pred_ft.predict(
                        multimask_output=False, point_coords=pt_c, point_labels=pt_l
                    )
                cv2.imwrite(str(dir_ft_cvc / f"{n}.png"), (masks[0] > 0).astype(np.uint8) * 255)

    if pred_ft is not None:
        del pred_ft
        torch.cuda.empty_cache()

# =============================================================================
# 3. INFERÊNCIA NO CVC-CLINICDB — U-NET (3 SEEDS, COM thr DA VALIDAÇÃO)
# =============================================================================
for s in (1, 2, 3):
    r_unet = next((BASE / "runs_UNet").glob(f"*seed{s}"))
    dir_u_cvc = r_unet / "pred_cvc"
    forcar_u = FORCAR_TUDO or FORCAR_UNET
    if not forcar_u and (dir_u_cvc / f"{nomes_cvc[-1]}.png").exists():
        print(f"✓ U-Net (Seed {s}) CVC já calculado.")
        continue

    thr_u = float(np.load(BASE / f"unet_val_seed{s}.npz")["thr"])
    keras_u = list(BASE.glob(f"*unet*seed{s}*.keras")) + list(r_unet.glob("*.keras"))
    assert keras_u, f"Modelo Keras da U-Net (Seed {s}) não encontrado."
    import tf_keras

    print(f"▶ A correr U-Net (Seed {s}, thr={thr_u}) no CVC-ClinicDB...", flush=True)
    if dir_u_cvc.exists():
        shutil.rmtree(dir_u_cvc)
    dir_u_cvc.mkdir(parents=True, exist_ok=True)

    u_model = tf_keras.models.load_model(str(keras_u[0]), compile=False)
    in_h, in_w = u_model.input_shape[1], u_model.input_shape[2]
    batch_imgs = [
        cv2.resize(ler_imagem(n, CVC_FAST), (in_w, in_h)).astype(np.float32) / 255.0
        for n in nomes_cvc
    ]
    preds_cvc = u_model.predict(np.stack(batch_imgs, axis=0), batch_size=16, verbose=0)

    for n, pr in zip(nomes_cvc, preds_cvc):
        gt = ler_mascara(n, CVC_FAST)
        prob_full = cv2.resize(
            np.squeeze(pr), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        pr_bin = (prob_full > thr_u).astype(np.uint8) * 255
        cv2.imwrite(str(dir_u_cvc / f"{n}.png"), pr_bin)

    del u_model
    tf_keras.backend.clear_session()

# =============================================================================
# 4. AVALIAR AS MÁSCARAS EM PARALELO E GERAR A TABELA FINAL DO CVC-CLINICDB
# =============================================================================
gts_cvc = {}
for n in nomes_cvc:
    gt = ler_mascara(n, CVC_FAST)
    if (gt > 0).sum() > 0:
        gts_cvc[n] = gt
print(f"\n✅ {len(gts_cvc)} máscaras GT do CVC carregadas em RAM.", flush=True)


def avaliar_cvc_pasta_rapido(pred_dir: Path):
    ficheiros = {f: pred_dir / f for f in os.listdir(pred_dir) if f.endswith(".png")}

    def proc_um(item):
        n, gt = item
        f = ficheiros.get(f"{n}.png") or ficheiros.get(f"{n}_p0.png")
        if f is None:
            f = next((p for nome_f, p in ficheiros.items() if n in nome_f), None)
            if f is None:
                return None
        p = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if p is None:
            return None
        if p.shape != gt.shape:
            p = cv2.resize(p, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        p_bin = (p > 127).astype(np.uint8)
        d = dice(gt, p_bin)
        u = iou(gt, p_bin)
        a_r, a_p = (gt > 0).sum(), p_bin.sum()
        err = abs(float(a_p) - float(a_r)) / (float(a_r) + 1e-7)
        return d, u, err

    with ThreadPoolExecutor(max_workers=12) as ex:
        res = [r for r in ex.map(proc_um, gts_cvc.items()) if r is not None]

    arr = np.array(res, dtype=np.float64)
    return (
        float(arr[:, 0].mean()),
        float(arr[:, 1].mean()),
        float(arr[:, 2].mean()),
        float((arr[:, 0] < 0.5).mean() * 100.0),
    )


kvasir_csv = OUT / "tabela_global_3seeds.csv"
kvasir_map = {}
if kvasir_csv.exists():
    df_kv = pd.read_csv(kvasir_csv)
    kvasir_map = {
        r["Modelo"]: float(r["mDice (Média ± DP)"].split("±")[0])
        for _, r in df_kv.iterrows()
    }

modelos_cvc = [
    ("U-Net",           [next((BASE / "runs_UNet").glob(f"*seed{s}")) / "pred_cvc" for s in (1, 2, 3)]),
    ("SAM1 ZS (Caixa)", [BASE / "runs_SAMsemfine/sam_vit_b/caixa_j0.0/cvc"]),
    ("SAM1 ZS (Ponto)", [BASE / "runs_SAMsemfine/sam_vit_b/ponto_centro_j0.0/cvc"]),
    ("SAM2 ZS (Caixa)", [BASE / "runs_SAMsemfine/sam2_hiera_s/caixa_j0.0/cvc"]),
    ("SAM2 ZS (Ponto)", [BASE / "runs_SAMsemfine/sam2_hiera_s/ponto_centro_j0.0/cvc"]),
    ("SAM1 FT (Caixa)", [BASE / f"runs_SAMfine/sam_vit_b_caixa/split{s}_seed{s}/pred_cvc" for s in (1, 2, 3)]),
    ("SAM1 FT (Ponto)", [BASE / f"runs_SAMfine/sam_vit_b_ponto_centro/split{s}_seed{s}/pred_cvc" for s in (1, 2, 3)]),
    ("SAM2 FT (Caixa)", [BASE / f"runs_SAMfine/sam2_hiera_s_caixa/split{s}_seed{s}/pred_cvc" for s in (1, 2, 3)]),
    ("SAM2 FT (Ponto)", [BASE / f"runs_SAMfine/sam2_hiera_s_ponto_centro/split{s}_seed{s}/pred_cvc" for s in (1, 2, 3)]),
]

cvc_rows = []
for nome_m, dirs_m in modelos_cvc:
    dirs_ok = [d for d in dirs_m if d.exists() and any(d.glob("*.png"))]
    if not dirs_ok:
        continue
    print(f"▶ A calcular métricas: {nome_m} ({len(dirs_ok)} pastas)...", flush=True)
    md_s, mi_s, er_s, fg_s = [], [], [], []
    for d_path in dirs_ok:
        md, mi, er, fg = avaliar_cvc_pasta_rapido(d_path)
        md_s.append(md); mi_s.append(mi); er_s.append(er); fg_s.append(fg)
    dp_md = np.std(md_s, ddof=1) if len(md_s) > 1 else 0.0
    dp_mi = np.std(mi_s, ddof=1) if len(mi_s) > 1 else 0.0
    md_med = float(np.mean(md_s))
    row = {
        "Modelo": nome_m,
        "CVC mDice (Média ± DP)": f"{md_med:.4f} ± {dp_md:.4f}",
        "CVC mIoU (Média ± DP)": f"{np.mean(mi_s):.4f} ± {dp_mi:.4f}",
        "Erro Rel. Área (%)": f"{np.mean(er_s)*100:.2f}%",
        "Falhas Dice<0.5 (%)": f"{np.mean(fg_s):.1f}%",
        "_ord": md_med,
    }
    if nome_m in kvasir_map:
        row["Δ mDice (CVC - Kvasir)"] = round(md_med - kvasir_map[nome_m], 4)
    cvc_rows.append(row)

df_cvc = pd.DataFrame(cvc_rows).sort_values("_ord", ascending=False).drop(columns=["_ord"])
df_cvc.drop(columns=["Δ mDice (CVC - Kvasir)"], errors="ignore").to_csv(
    OUT / "tabela_generalizacao_cvc_3seeds.csv", index=False
)
df_cvc.to_csv(OUT / "tabela_5_generalizacao_cvc_completa.csv", index=False)

print("\n=== GENERALIZAÇÃO EXTERNA — CVC-CLINICDB (612 IMAGENS, CORRIGIDA) ===")
print(df_cvc.to_string(index=False))
print(f"\n✅ Tabelas guardadas em: {OUT}")