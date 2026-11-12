# G1 — Segmentação de pólipos em colonoscopia com modelos fundacionais (Kvasir-SEG)
Projeto P2 · Deep Learning Aplicado · 2026/2027 

**Elementos:** Eliana Monteiro, Miguel Carreira, Sara Costa, José Peixoto, Daniel Moreira

---

## 1. Problema

Na colonoscopia, a forma e o tamanho de um pólipo condicionam a decisão de o remover e a técnica a usar. Desenhar a máscara exata do pólipo numa imagem é o primeiro passo para medir a área, acompanhar a lesão e apoiar a decisão do médico.

- **Utilizador:** gastroenterologista, durante a revisão de imagens de colonoscopia.
- **Decisão apoiada:** caracterizar a forma e o tamanho do pólipo para apoiar a remoção e a técnica; registar a lesão para seguimento.
- **Tarefa:** segmentação binária de pólipos (pólipo vs. fundo).

**Pergunta central:** um modelo que nunca viu uma colonoscopia (SAM/SAM 2, com prompts) aproxima-se, iguala ou supera uma U-Net treinada em 800 imagens do domínio? Com que tipo de prompt, com que robustez à imprecisão do utilizador e em que casos falha?

## 2. Dados

| Conjunto | Utilização | Conteúdo | Licença / citação |
|---|---|---|---|
| **Kvasir-SEG** | Treino, validação e teste | 1000 imagens de colonoscopia com máscara binária e caixa envolvente; resolução variável (de 332×487 a 1920×1072) | Uso em investigação e ensino, **não comercial**. Jha et al. (2020) |
| **CVC-ClinicDB** | Teste externo (mudança de distribuição) | 612 fotogramas de 31 sequências, com máscara | Público. Bernal et al. (2015) |

- **Divisão:** 80/10/10 (800/100/100), repetida com 3 seeds (`data/splits_seed{1,2,3}.json`, versionados).
- **CVC-ClinicDB** nunca é usado para treinar nem para escolher nada (modelo, threshold, máscara).
- Os dados em si ficam fora do git (ver `.gitignore`); só as divisões e a tabela de resoluções são versionadas.

**Limitação declarada:** o Kvasir-SEG não traz identificação do doente. Imagens do mesmo doente podem cair em treino e em teste, o que tende a inflacionar as métricas. A unidade de agrupamento possível é a imagem. A limitação é mitigada com várias seeds e divisões e com o teste externo no CVC-ClinicDB.

## 3. Métodos

| Via | Método |
|---|---|
| Baseline | U-Net com encoder pré-treinado em ImageNet (ex.: ResNet-34), Dice+BCE, augmentation, 352×352, 3 seeds |
| Método avançado | SAM (ViT-B) e SAM 2 (Hiera-S) com prompt de ponto ou de caixa, zero-shot |
| Método avançado (afinação leve) | Afinação do descodificador de máscara do SAM (encoder congelado, embeddings pré-calculados) |
| Pipeline automático | U-Net → caixa envolvente → SAM, sem intervenção do utilizador |

Os prompts "oráculo" (derivados da máscara real) são um **limite superior** e são identificados como tal. Os cenários realistas são o prompt com jitter e o pipeline automático.

## 4. Avaliação

- **Métrica principal:** Dice médio por imagem (mDice), com IC 95% por bootstrap de imagens, calculado na resolução original da máscara.
- **Métricas secundárias:** mIoU; mDice por subgrupo (pólipos pequenos < 2% da imagem, brilho especular, bordos pouco nítidos); erro relativo da área; desvio do Dice entre prompts com jitter; tempo por imagem.
- **Comparação justa:** mesmas imagens de teste, mesma divisão, mesma métrica, calculada pelo mesmo código (`src/avaliar.py`), várias seeds.
- **Teste externo:** todas as condições avaliadas no CVC-ClinicDB sem re-treino.

O protocolo experimental completo (hipóteses H1–H5, critérios de seleção de modelo, análises previstas) está em [`PROTOCOLO.md`](PROTOCOLO.md).

## 5. Estrutura do repositório

```
DL_P2_TP1-G<n>/
  README.md        # problema, dados, como executar, estado das vias
  AI_LOG.md        # registo do uso de IA (todos os membros)
  PROTOCOLO.md     # protocolo experimental, fechado antes de tocar no teste
  configs/         # um YAML por experiência: modelo, prompt, seed, split
  data/            # fora do git; splits_seed{1,2,3}.json versionados
  src/             # dados.py, unet.py, sam_prompts.py, sam_finetune.py,
                   # avaliar.py, prompts.py, area.py
  notebooks/       # exploração; nenhum resultado final nasce aqui
  runs/            # logs, checkpoints e máscaras previstas; fora do git
  app/app.py       # interface Gradio: imagem + clique/caixa -> máscara + área
  paper/           # resultados.csv, leituras/, figuras/, tabelas/, manuscrito/
  requirements.txt
```

## 6. Como executar

> A preencher à medida que o código for sendo escrito.

```bash
# 1. Clonar o repositório e instalar as dependências
pip install -r requirements.txt

# 2. Colocar o Kvasir-SEG e o CVC-ClinicDB em data/ (não versionados)

# 3. Lançar a aplicação (a partir do protótipo P2-D3)
python app/app.py
```

A aplicação recebe uma imagem de colonoscopia e um clique no pólipo, uma caixa desenhada ou o modo automático. Devolve a máscara sobreposta, a área estimada (píxeis e % da imagem), o modelo usado e o tempo.

> **Aviso:** apoio à decisão; não substitui a avaliação clínica.

## 7. Papéis no grupo

| Papel | Responsável por | Elemento |
|---|---|---|
| A1 Dados | Divisões, CVC-ClinicDB, `PROTOCOLO.md`, `resultados.csv`, aplicação Gradio | Eliana Monteiro |
| A2 Modelos | U-Net com encoder pré-treinado, 3 seeds, threshold na validação | Miguel Carreira |
| A3 Porta-voz | SAM e SAM 2 com prompts de ponto e caixa, oráculo e jitter; seminário | Sara Costa |
| A4 Aplicação | Interface (Gradio/Streamlit) e integração do modelo | José Peixoto |
| A5 Qualidade e IA | `avaliar.py`, bootstrap, subgrupos, piores casos, estabilidade a prompts | Daniel Moreira |

## 8. Estado das vias

| Via | Estado |
|---|---|
| Repositório e estrutura | ✅ criado |
| Proposta (P2-D1) | ✅ criado |
| Divisões com 3 seeds (A1) | ✅ criado |
| U-Net baseline (A2) | ☐ |
| SAM / SAM 2 zero-shot (A3) | ☐ |
| SAM afinado (A4) | ☐ |
| `avaliar.py` e subgrupos (A5) | ☐ |
| Aplicação Gradio | ☐ |
| Teste externo (CVC-ClinicDB) | ☐ |

## 9. Calendário

| Entrega | Data |
|---|---|
| P2-D1 — proposta | 28/10 |
| P2-D2 — seminário (tema A) | 17/11 |
| P2-D3 — protótipo (tag `v1-D3`) | 02/12 |
| P2-D4 | 16/12 |
| Relatório, repositório e `AI_LOG.md` (tag `v2-final`) | 04/01 |
| P2-D5 — apresentação / defesa individual | 05/01 e 06/01 |

## 10. Referências

- Jha, D. et al. (2020). *Kvasir-SEG: A Segmented Polyp Dataset.* (dados principais)
- Bernal, J. et al. (2015). CVC-ClinicDB. (teste externo)
- Ronneberger, O. et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.*
- Kirillov, A. et al. (2023). *Segment Anything.*
- Ravi, N. et al. (2024). *SAM 2: Segment Anything in Images and Videos.*
- Ma, J. et al. (2024). *Segment Anything in Medical Images* (MedSAM).
- Fan, D.-P. et al. (2020). *PraNet: Parallel Reverse Attention Network for Polyp Segmentation.*
- Maier-Hein, L. et al. (2024). *Metrics Reloaded.*

A lista completa fica em `paper/referencias.bib`.

## 11. Licenças e uso de IA

- Os conjuntos Kvasir-SEG e CVC-ClinicDB são para uso em investigação e ensino, **não comercial**. Jha et al. (2020) e Bernal et al. (2015) são citados sempre que os dados são mencionados.
- O SAM e o SAM 2 têm as respetivas licenças da Meta; verificar antes de qualquer utilização fora do contexto académico.
- O uso de assistentes de IA está registado em [`AI_LOG.md`](AI_LOG.md).
