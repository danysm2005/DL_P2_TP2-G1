# G1 — Segmentação de pólipos em colonoscopia com modelos fundacionais (Kvasir-SEG)
Projeto P2 · Aprendizagem Profunda Aplicada · 2026/2027 

**Elementos:** Eliana Monteiro, Sara Costa, José Peixoto, Daniel Moreira

---

## 1. Problema

Na colonoscopia, a forma e o tamanho de um pólipo condicionam a decisão de o remover e a técnica a usar. Desenhar a máscara exata do pólipo numa imagem é o primeiro passo para medir a área, acompanhar a lesão e apoiar a decisão do médico.

- **Utilizador:** gastroenterologista, durante a revisão de imagens de colonoscopia.
- **Decisão apoiada:** caracterizar a forma e o tamanho do pólipo para apoiar a remoção e a técnica; registar a lesão para seguimento.
- **Tarefa:** segmentação binária de pólipos (pólipo vs. fundo).

**Questão central:** Um modelo que nunca foi treinado especificamente com imagens de colonoscopia (SAM/SAM 2), orientado por prompts, consegue aproximar-se, igualar ou superar uma U-Net treinada com 800 imagens do domínio? Como varia o desempenho consoante o tipo de indicação fornecida pelo utilizador? Até que ponto se mantém quando essas indicações são imprecisas? E em que situações falha?

## 2. Dados

| Conjunto | Utilização | Conteúdo | Licença / citação |
|---|---|---|---|
| **Kvasir-SEG** | Treino, validação e teste | 1000 imagens de colonoscopia com máscara binária e caixa envolvente; resolução variável (de 332×487 a 1920×1072) | Uso em investigação e ensino, não comercial. Jha et al. (2020) |
| **CVC-ClinicDB** | Teste externo (mudança de distribuição) | 612 fotogramas extraídos de 31 sequências, com máscara | Público. Bernal et al. (2015) |

- **Divisão:** 80/10/10 (800/100/100), repetida com 3 seeds (`data/splits_seed{1,2,3}.json`, versionados).
- **CVC-ClinicDB** reservado à avaliação externa, não sendo utilizado no treino ou na seleção e afinação dos modelos.
- Os dados em si ficam fora do git (ver `.gitignore`); só as divisões e a tabela de resoluções são versionadas.

**Limitação declarada:** o Kvasir-SEG não traz identificação do doente. Imagens do mesmo doente podem cair em treino e em teste, o que tende a inflacionar as métricas. A unidade de agrupamento possível é a imagem. A limitação é mitigada com várias seeds e divisões e com o teste externo no CVC-ClinicDB.

## 3. Métodos

| Modelo | Método |
|---|---|
| Baseline | U-Net com encoder pré-treinado em ImageNet (ResNet-34), Dice+BCE, augmentation, 352×352, 3 seeds |
| Método avançado | SAM (ViT-B) e SAM 2 (Hiera-S) com prompt de ponto ou de caixa, zero-shot |
| Método avançado (afinação leve) | Afinação do descodificador de máscara do SAM (encoder congelado, embeddings pré-calculados) |

Os prompts "oráculo" (derivados da máscara real) são um limite superior e são identificados como tal. Os cenários realistas são o prompt com jitter e o pipeline automático.

## 4. Avaliação

- **Métrica principal:** Dice médio por imagem (mDice), com IC 95% por bootstrap de imagens, calculado na resolução original da máscara.
- **Métricas secundárias:** mIoU; mDice por subgrupo (pólipos pequenos < 2% da imagem, brilho especular, bordos pouco nítidos); erro relativo da área; desvio do Dice entre prompts com jitter; tempo por imagem.
- **Comparação justa:** mesmas imagens de teste, mesma divisão, mesma métrica, calculada pelo mesmo código (`src/avaliar.py`), várias seeds.
- **Teste externo:** todas as condições avaliadas no CVC-ClinicDB sem re-treino.

O protocolo experimental completo (hipóteses H1–H5, critérios de seleção de modelo, análises previstas) está em [`PROTOCOLO.md`](PROTOCOLO.md).

## 5. Estrutura do Repositório

```
DL_P2_TP1-G1/
  README.md        # problema, dados, como executar, estado das vias
  AI_LOG.md        # registo do uso de IA (todos os membros)
  PROTOCOLO.md     # protocolo experimental, fechado antes de tocar no teste
  configs/         # um YAML por experiência: modelo, prompt, seed, split
  data/            # fora do git; splits_seed{1,2,3}.json versionados
  src/             # dados.py, sam_prompts.py, sam_finetune.py,
                   # avaliar.py, sam_ft_multi.py, __init__.py
  notebooks/       # 1_unet_sam_analise.py, 2_jitter.py, 3_cvc.py (exportados do Colab)
  runs/            # logs, checkpoints e máscaras previstas; fora do git
  app/app.py       # interface Gradio: imagem + clique/caixa -> máscara + área
  paper/           # resultados.csv, figuras/ e tabelas/ com os reultados finais
  requirements.txt
```

## 6. Como executar

O código foi corrido no Google Colab (GPU T4), com os dados e os resultados numa pasta `P2_unet` da Google Drive. Os ficheiros em `notebooks/` são exportações do Colab e contêm comandos `!pip` e `drive.mount`: devem ser copiados célula a célula para um Colab, não corridos com `python`.

```text
1. Colab novo com GPU. Instalar: pip install -r requirements.txt
2. Drive: pasta MyDrive/P2_unet com o código de src/, os splits e os dados:
   - Kvasir-SEG: MyDrive/archive.zip
   - CVC-ClinicDB: P2_unet/data/cvc-clinicdb/{images,masks}
3. Ordem de execução:
   notebooks/1_unet_sam_analise.py   splits, U-Net, SAM zero-shot e afinado, análises e tabelas
   notebooks/2_jitter.py             robustez ao jitter (zero-shot e afinado)
   notebooks/3_cvc.py                teste externo no CVC-ClinicDB
4. Seeds 2 e 3: na U-Net muda-se SEED (célula de treino); no SAM afinado
   usam-se --split e --seed em src/sam_ft_multi.py.
5. Os resultados ficam em P2_unet/resultados_finais_relatorio (cópia em paper/).
```

**Notas**
- Seed de treino = seed de divisão: a seed, s, usa `data/splits_seed{s}.json`. Em `runs_UNet/split1_seed{s}`, o «split1» é só parte do nome da pasta.
- Os pesos (`*.keras`, `*.pt`, `*.pth`) e as probabilidades (`*.npz`) não estão no git; ficam na Drive do grupo.
- Aplicação: `app/app.py` usa `src/inferencia.py`, que lê os pesos da Drive (variável de ambiente `P2_BASE`).

> **Aviso:** apoio à decisão; não substitui a avaliação clínica.

## 6.1. Resultados principais (Kvasir-SEG, teste, 3 seeds)

| Modelo | mDice | CVC-ClinicDB mDice |
|---|---|---|
| SAM 2 afinado (caixa) | 0.9372 ± 0.0038 | 0.9244 ± 0.0039 |
| SAM 1 afinado (caixa) | 0.9231 ± 0.0021 | 0.9103 ± 0.0057 |
| SAM 2 afinado (ponto) | 0.9175 ± 0.0159 | 0.8719 ± 0.0038 |
| U-Net | 0.8995 ± 0.0145 | 0.8361 ± 0.0101 |
| SAM 1 afinado (ponto) | 0.8918 ± 0.0123 | 0.7973 ± 0.0115 |
| SAM 2 zero-shot (caixa) | 0.8795 ± 0.0180 | 0.9211 |
| SAM 1 zero-shot (caixa) | 0.8624 ± 0.0124 | 0.8968 |
| SAM 2 zero-shot (ponto) | 0.7466 ± 0.0411 | 0.7361 |
| SAM 1 zero-shot (ponto) | 0.5288 ± 0.0409 | 0.4467 |

Os prompts de caixa e de ponto vêm da máscara real (oráculo): são um limite superior. Intervalos de confiança, diferenças emparelhadas, jitter, subgrupos e piores casos estão na secção 13 do [`PROTOCOLO.md`](PROTOCOLO.md).

## 7. Papéis no grupo

| Papel | Responsável por | Elemento |
|---|---|---|
| A1 Dados | Divisões, CVC-ClinicDB, `PROTOCOLO.md`, `resultados.csv`| Eliana Monteiro |
| A2 Modelos | U-Net com encoder pré-treinado, 3 seeds, threshold na validação | Daniel Moreira |
| A3 Porta-voz | SAM e SAM 2 com prompts de ponto e caixa, oráculo e jitter; seminário | Sara Costa |
| A4 Aplicação | Interface (Gradio/Streamlit) e integração do modelo | José Peixoto |
| A5 Qualidade e IA | `avaliar.py`, bootstrap, subgrupos, piores casos, estabilidade a prompts | Daniel Moreira |

## 8. Estado das vias

| Via | Estado |
|---|---|
| Repositório e estrutura | ✅ criado |
| Proposta (P2-D1) | ✅ criado |
| Divisões com 3 seeds | ✅ criado |
| U-Net baseline | ✅ criado |
| SAM / SAM 2 zero-shot | ✅ criado |
| SAM afinado | ✅ criado |
| `avaliar.py`, subgrupos,  piores casos, calibração | ✅ criado |
| Robustez ao jitter | ✅ criado |
| Grad-CAM e curva de deleção na U-Net | ✅ criado |
| Aplicação Gradio | ✅ criado |
| Teste externo (CVC-ClinicDB) | ✅ criado |

## 9. Calendário

| Entrega | Data |
|---|---|
| P2-D1 — Proposta do Projeto | 28/10 |
| P2-D2 — Seminário I (tema A2) | 17/11 |
| P2-D3 — Protótipo (tag `v1-D3`) | 02/12 |
| P2-D4 — Resultados e Discussão| 16/12 |
| Relatório, repositório e `AI_LOG.md` (tag `v2-final`) | 04/01 |
| P2-D5 — apresentação / defesa individual | 05/01 e 07/01 |

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

- Os conjuntos Kvasir-SEG e CVC-ClinicDB são para uso em investigação e ensino, não comercial. Jha et al. (2020) e Bernal et al. (2015) são citados sempre que os dados são mencionados.
- O SAM e o SAM 2 (código e pesos) estão sob licença Apache 2.0 (Meta); confirmar nos repositórios oficiais antes de qualquer utilização fora do contexto académico.
- O uso de assistentes de IA está registado em [`AI_LOG.md`](AI_LOG.md).
