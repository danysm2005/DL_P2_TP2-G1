"""Dados: leitura, divisões 80/10/10 com hash, Dataset da U-Net (A1/A2)."""
import hashlib
import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

KVASIR = next(Path("/content/kvasir_dataset").rglob("images")).parent   # images/ e masks/
CVC = Path("data/cvc-clinicdb")    # converter para o mesmo layout: images/ e masks/
SIZE = 352
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _achar(pasta, nome):
    ficheiros = list(pasta.glob(f"{nome}.*"))
    assert ficheiros, f"não encontrei {nome} em {pasta}"
    return ficheiros[0]


def listar(root=KVASIR):
    return sorted(p.stem for p in (root / "images").iterdir() if p.is_file())


def ler_imagem(nome, root=KVASIR):
    img = cv2.imread(str(_achar(root / "images", nome)))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def ler_mascara(nome, root=KVASIR):
    m = cv2.imread(str(_achar(root / "masks", nome)), cv2.IMREAD_GRAYSCALE)
    return (m > 127).astype(np.uint8)  # as máscaras .jpg têm ruído de compressão


def criar_divisao(seed, out=None):
    nomes = listar()
    n = len(nomes)
    p = np.random.default_rng(seed).permutation(n)
    nomes = [nomes[i] for i in p]
    a, b = int(0.8 * n), int(0.9 * n)
    d = {"seed": seed, "train": sorted(nomes[:a]), "val": sorted(nomes[a:b]),
         "test": sorted(nomes[b:])}
    conteudo = json.dumps({k: d[k] for k in ("train", "val", "test")}, sort_keys=True)
    d["hash"] = hashlib.sha256(conteudo.encode()).hexdigest()
    out = Path(out or f"data/splits_seed{seed}.json")
    out.write_text(json.dumps(d, indent=1))
    return d


def carregar_divisao(seed):
    d = json.loads(Path(f"/content/drive/MyDrive/P2_unet/splits/splits_seed{seed}.json").read_text())
    if "hash" not in d:  # os splits do notebook não têm hash: calcula-o
        conteudo = json.dumps({k: d[k] for k in ("train", "val", "test")}, sort_keys=True)
        d["hash"] = hashlib.sha256(conteudo.encode()).hexdigest()
    return d


def transformacoes(treino):
    aug = []
    if treino:
        aug = [A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90(),
               A.Affine(scale=(0.85, 1.15), rotate=(-25, 25),
                        translate_percent=(-0.05, 0.05), p=0.5),
               A.ColorJitter(0.2, 0.2, 0.2, 0.02, p=0.5)]
    return A.Compose([A.Resize(SIZE, SIZE)] + aug +
                     [A.Normalize(mean=tuple(MEAN), std=tuple(STD)), ToTensorV2()])


class KvasirDataset(Dataset):
    def __init__(self, nomes, treino, root=KVASIR):
        self.nomes, self.root, self.t = nomes, root, transformacoes(treino)

    def __len__(self):
        return len(self.nomes)

    def __getitem__(self, i):
        n = self.nomes[i]
        o = self.t(image=ler_imagem(n, self.root), mask=ler_mascara(n, self.root))
        return o["image"], o["mask"].float().unsqueeze(0)
