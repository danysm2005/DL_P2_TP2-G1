#gradio: U-Net, SAM e SAM 2 (com e sem afinação), com Dice, IoU, HD95 e tempo em segundos
import inspect
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
from scipy.ndimage import distance_transform_edt

# app/app.py -> a raiz do projeto (pasta onde está src/) é o pai da pasta app/
try:
    RAIZ = Path(__file__).resolve().parents[1]
except NameError:                                   # se correres num notebook (sem __file__)
    RAIZ = Path("/content/drive/MyDrive/P2_app")
sys.path.insert(0, str(RAIZ))

from src.inferencia import (MODELOS, detalhe_modelo, metricas_modelo, modelos_simulados,
                            pre_carregar, predict)

# --------------------------------------------------------------------------- #
# Estilo
# --------------------------------------------------------------------------- #
TEMA = gr.themes.Soft(
    primary_hue="teal", secondary_hue="cyan", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_lg,
)

CSS = """
.gradio-container { max-width: 1200px !important; margin: auto; }
footer { display: none !important; }
.hero { background: linear-gradient(135deg, #0f766e 0%, #0891b2 100%);
  color: #fff; padding: 28px 32px; border-radius: 16px; margin-bottom: 12px; }
.hero h1 { margin: 0 0 6px 0; font-size: 1.9rem; font-weight: 700; color: #fff; }
.hero p  { margin: 0; opacity: .92; font-size: 1rem; color: #fff; }
.aviso { border-left: 5px solid #f59e0b; background: rgba(245,158,11,.12);
  padding: 12px 16px; border-radius: 10px; font-size: .92rem; margin-bottom: 8px; }
.grelha  { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
.grelha3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.seccao { font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; opacity: .6; margin: 14px 0 6px; }
.cartao { background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
  border-radius: 14px; padding: 14px 16px; }
.cartao .rotulo { font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; opacity: .65; }
.cartao .valor  { font-size: 1.45rem; font-weight: 700; margin-top: 2px; }
.cartao .sub    { font-size: .78rem; opacity: .65; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: .85rem; font-weight: 600; }
.ok    { background: rgba(16,185,129,.18); color: #059669; }
.warn  { background: rgba(245,158,11,.20); color: #b45309; }
.vazio { color: var(--body-text-color-subdued); text-align: center; padding: 32px 0; }
.rodape { font-size: .78rem; opacity: .65; text-align: center; margin-top: 10px; }
"""

HERO = """
<div class="hero">
  <h1>🩺 Segmentação de Pólipos</h1>
  <p>Colonoscopia · Kvasir-SEG · A U-Net segmenta sozinha; o SAM e o SAM 2 (com e sem afinação) são guiados por clique ou caixa.</p>
</div>
"""

AVISO = """
<div class="aviso">
  <b>⚠️ Protótipo académico — apoio à decisão; não usar para decisões clínicas reais.</b>
  A área em mm² exige calibração, não disponível.
</div>
"""

RODAPE = ('<div class="rodape">Dados: Kvasir-SEG, Jha et al. (2020), uso em investigação e ensino, '
          'não comercial · Teste externo: CVC-ClinicDB, Bernal et al. (2015)</div>')

ESTADO_INICIAL = '<div class="vazio">Os resultados aparecem aqui depois de segmentar.</div>'


# --------------------------------------------------------------------------- #
# Lógica e Métricas
# --------------------------------------------------------------------------- #
MODOS = ["Ponto (1 clique)", "Caixa (2 cliques)"]
LIMIAR_ESTABILIDADE = 0.80


def precisa_prompt(modelo):
    return modelo != "U-Net"


def dica(modelo):
    if precisa_prompt(modelo):
        return msg("👆 Clique no pólipo (ou nos dois cantos de uma caixa) na imagem.")
    return msg("A U-Net não usa prompts: carregue uma imagem e clique em «Segmentar».")


def dice(a, b, eps=1e-7):
    return float((2 * (a & b).sum() + eps) / (a.sum() + b.sum() + eps))


def iou(a, b, eps=1e-7):
    return float(((a & b).sum() + eps) / ((a | b).sum() + eps))


def _contorno(m):
    p = np.pad(m, 1)
    erosao = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~erosao


def hausdorff(a, b):
    """Devolve (HD95, HD_max) em píxeis entre os contornos de duas máscaras binárias."""
    if not a.any() or not b.any():
        return None, None
    ca, cb = _contorno(a), _contorno(b)
    if not ca.any() or not cb.any():
        return None, None
    dist_a_para_b = distance_transform_edt(~cb)[ca]
    dist_b_para_a = distance_transform_edt(~ca)[cb]
    todas = np.concatenate([dist_a_para_b, dist_b_para_a])
    return float(np.percentile(todas, 95)), float(todas.max())


def preparar_gt(gt, forma):
    """Devolve (máscara booleana ou None, aviso)."""
    if gt is None:
        return None, ""
    g = np.asarray(gt)
    if g.ndim == 3:
        g = g[..., 0]
    aviso = ""
    if g.shape != tuple(forma):
        if abs((g.shape[1] / g.shape[0]) / (forma[1] / forma[0]) - 1) > 0.02:
            aviso = "A máscara tem proporções diferentes da imagem: confirme que é a máscara desta imagem."
        from PIL import Image
        g = np.array(Image.fromarray(g.astype(np.uint8)).resize((forma[1], forma[0]), Image.NEAREST))
    gb = g > (0 if g.max() <= 1 else 127)          # aceita 0/1 e 0/255
    if not gb.any():
        return None, "A máscara real está vazia: não é possível calcular Dice, IoU e HD."
    return gb, aviso


def sobrepor(imagem, mascara, cor=(239, 68, 68), alpha=0.5):
    out = imagem.astype(np.float32).copy()
    m = mascara > 0
    out[m] = (1 - alpha) * out[m] + alpha * np.array(cor, dtype=np.float32)
    out[_contorno(m)] = (255, 255, 255)
    return out.clip(0, 255).astype(np.uint8)


def marcar(img, ponto):
    out = img.copy()
    x, y = map(int, ponto)
    out[max(y - 5, 0):y + 6, max(x - 5, 0):x + 6] = (239, 68, 68)
    return out


def cartao(rotulo, valor, sub=""):
    return (f'<div class="cartao"><div class="rotulo">{rotulo}</div>'
            f'<div class="valor">{valor}</div><div class="sub">{sub}</div></div>')


def msg(texto):
    return f'<div class="vazio">{texto}</div>'


def fmt(v, casas=3):
    return "—" if v is None else f"{v:.{casas}f}"


def resultado(img, prompt, modelo, gt=None):
    t0 = time.perf_counter()
    masks, scores = predict(img, prompt, modelo)
    mask = masks[int(np.argmax(scores))]
    tempo = time.perf_counter() - t0

    px = int(mask.sum())
    pct = 100 * px / mask.size

    e_unet = modelo == "U-Net"
    sem_candidatas = e_unet or modelo.endswith("afinado")      # 1 só máscara repetida 3x
    estab = float(np.mean([dice(masks[i], masks[j]) for i, j in [(0, 1), (0, 2), (1, 2)]]))
    estab_txt = "—" if sem_candidatas else f"{estab:.2f}"
    estab_sub = "1 só máscara (sem alternativas)" if sem_candidatas else "Dice entre 3 máscaras candidatas"

    simulado = modelo in modelos_simulados()
    nome = modelo + (" · SIMULADO" if simulado else "")
    sub_modelo = "checkpoint não encontrado" if simulado else detalhe_modelo(modelo)

    if simulado:
        estado = '<span class="badge warn">⚠️ Resultado simulado (sem modelo real)</span>'
    elif px == 0:
        estado = '<span class="badge warn">🔴 Máscara vazia</span>'
    elif not sem_candidatas and estab < LIMIAR_ESTABILIDADE:
        estado = '<span class="badge warn">🟠 Máscara instável</span>'
    else:
        estado = '<span class="badge ok">🟢 Segmentação concluída</span>'

    # Métricas DESTA imagem (quando a máscara real é carregada)
    g, aviso_gt = preparar_gt(gt, mask.shape)
    aviso_html = f'<div class="aviso">⚠️ {aviso_gt}</div>' if aviso_gt else ""
    d_img = None if g is None else dice(g, mask)
    i_img = None if g is None else iou(g, mask)
    hd95_img, hdmax_img = (None, None) if g is None else hausdorff(g, mask)
    sub_img = "vs máscara real" if g is not None else "carregue a máscara real"
    hd95_txt = "—" if hd95_img is None else f"{hd95_img:.1f} px"
    hd95_sub = f"HD máx: {hdmax_img:.1f} px" if hdmax_img is not None else sub_img

    # mDice/mIoU do modelo no conjunto de teste
    md, mi = metricas_modelo(modelo, prompt)
    sub_mod = "média no conjunto de teste" if md is not None else "não encontrado no ranking"

    tempo_txt = f"{tempo:.2f} s"
    tempo_sub = "por imagem" if e_unet else "por imagem (inclui o encoder na 1.ª vez)"

    html = f"""
    <div style="margin-bottom:4px">{estado}</div>{aviso_html}
    <div class="seccao">Segmentação</div>
    <div class="grelha">
      {cartao("Área estimada", f"{pct:.1f}%", f"{px:,} píxeis".replace(",", " "))}
      {cartao("Estabilidade", estab_txt, estab_sub)}
      {cartao("Tempo", tempo_txt, tempo_sub)}
      {cartao("Modelo", nome, sub_modelo)}
    </div>
    <div class="seccao">Qualidade (esta imagem)</div>
    <div class="grelha3">
      {cartao("Dice", fmt(d_img), sub_img)}
      {cartao("IoU", fmt(i_img), sub_img)}
      {cartao("HD95", hd95_txt, hd95_sub)}
    </div>
    <div class="seccao">Referência do modelo (teste)</div>
    <div class="grelha">
      {cartao("mDice (modelo)", fmt(md), sub_mod)}
      {cartao("mIoU (modelo)", fmt(mi), sub_mod)}
    </div>"""
    return sobrepor(img, mask), html


def ao_clicar(imagem, modo, modelo, canto, gt, evt: gr.SelectData):
    if imagem is None:
        return None, msg("⬆️ Carregue uma imagem primeiro."), None
    img = imagem[..., :3]

    x, y = evt.index
    if not precisa_prompt(modelo):
        return None, dica(modelo), None

    if modo.startswith("Ponto"):
        return (*resultado(img, {"ponto": (x, y)}, modelo, gt), None)

    if canto is None:
        return marcar(img, (x, y)), msg("Canto 1 registado. Clique no canto oposto."), (x, y)

    x0, y0 = canto
    caixa = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
    return (*resultado(img, {"caixa": caixa}, modelo, gt), None)


def sem_prompt(imagem, modelo, gt):
    if imagem is None:
        return None, msg("⬆️ Carregue uma imagem primeiro.")
    return resultado(imagem[..., :3], {"auto": True}, modelo, gt)


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
GRADIO6 = "theme" in inspect.signature(gr.Blocks.launch).parameters
ESTILO = dict(theme=TEMA, css=CSS)

with gr.Blocks(title="Segmentação de Pólipos", **({} if GRADIO6 else ESTILO)) as demo:
    gr.HTML(HERO)
    gr.HTML(AVISO)
    canto = gr.State(None)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            imagem_entrada = gr.Image(label="Imagem de colonoscopia (clique para marcar o pólipo)",
                                      type="numpy", sources=["upload"], height=360)
            mascara_real = gr.Image(label="Máscara real / Ground Truth (opcional: calcula Dice, IoU e HD95)",
                                    type="numpy", sources=["upload"], height=160)
            modelo = gr.Radio(MODELOS, value="SAM", label="Modelo")
            modo = gr.Radio(MODOS, value=MODOS[0], label="Tipo de prompt (SAM e SAM 2)")
            aviso_modo = gr.HTML(dica("SAM"))
            with gr.Row():
                btn_seg = gr.Button("Segmentar", variant="primary", scale=3, visible=False)
                btn_limpar = gr.ClearButton(value="Limpar", scale=1)

        with gr.Column(scale=1):
            imagem_saida = gr.Image(label="Máscara sobreposta", interactive=False, height=360)
            resultados = gr.HTML(ESTADO_INICIAL)

    gr.HTML(RODAPE)

    imagem_entrada.select(ao_clicar, [imagem_entrada, modo, modelo, canto, mascara_real],
                          [imagem_saida, resultados, canto])
    btn_seg.click(sem_prompt, [imagem_entrada, modelo, mascara_real], [imagem_saida, resultados])

    def ao_mudar_modelo(m):
        p = precisa_prompt(m)
        return gr.update(visible=p), gr.update(visible=not p), dica(m), None

    modelo.change(ao_mudar_modelo, modelo, [modo, btn_seg, aviso_modo, canto])
    btn_limpar.add([imagem_entrada, mascara_real, imagem_saida])
    btn_limpar.click(lambda: (ESTADO_INICIAL, None), None, [resultados, canto])
    imagem_entrada.upload(lambda: None, None, canto)
    modo.change(lambda: None, None, canto)


if __name__ == "__main__":
    pre_carregar()
    demo.launch(share=True, **(ESTILO if GRADIO6 else {}))
