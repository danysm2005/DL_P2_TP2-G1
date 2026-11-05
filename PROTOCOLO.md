# PROTOCOLO.md: protocolo experimental

Tema A · Segmentação de pólipos em colonoscopia (Kvasir-SEG) · Grupo `DL_P2_TP1-G<n>`

**Estado: RASCUNHO.** Fecha-se e versiona-se na PL09 (11/11), **antes de qualquer avaliação no conjunto de teste**. Depois de fechado, só se altera com nota datada na secção 12.

Campos marcados com **[A2]**, **[A3]**, **[A4]** ou **[A5]** são para o responsável confirmar ou completar.

---

## 1. Pergunta central e hipóteses

**Pergunta:** até onde chega um modelo fundacional (SAM / SAM 2) sem treino específico, comparado com uma U-Net treinada em 800 imagens do domínio?

| Id | Hipótese |
|---|---|
| H1 | O SAM zero-shot com caixa oráculo atinge um mDice não inferior ao da U-Net (IC da diferença inclui 0 ou é favorável ao SAM). |
| H2 | A afinação leve do descodificador supera o zero-shot e a U-Net, com IC da diferença a excluir 0. |
| H3 | A caixa é melhor do que o ponto e perde menos de 3 pontos de Dice com jitter de 10%. |
| H4 | O Dice cai nos pólipos pequenos (< 2% da imagem) e nas imagens com brilho especular, em todas as condições. |
| H5 | No CVC-ClinicDB a U-Net degrada-se mais do que o SAM zero-shot (mudança de distribuição). |

## 2. Dados

| Conjunto | Papel | Tamanho |
|---|---|---|
| Kvasir-SEG (Jha et al., 2020) | treino, validação, teste | 1000 pares imagem–máscara |
| CVC-ClinicDB (Bernal et al., 2015) | **teste externo apenas** | 612 fotogramas, 31 sequências |

**Verificação dos dados (24/10/2026, A1).**
- 1000 imagens e 1000 máscaras do Kvasir-SEG, com nomes coincidentes; **0 pares** com imagem e máscara de tamanhos diferentes.
- Largura entre 332 e 1920 e altura entre 352 e 1072, com **333 resoluções distintas** (a mais frequente, 622×530, em 78 imagens). Tabela completa em `resolucoes.csv`.
- Em 10 máscaras aleatórias (seed 0): valores de 0 a 255, com 13 a 16 níveis distintos por artefactos de compressão JPEG (as máscaras são `.jpg`) e 100% dos píxeis a menos de 10 níveis de 0 ou 255.
- Área do pólipo nessas 10 máscaras: entre 1,2% e 50,7% da imagem (só 1 em 10 abaixo de 2%; é uma amostra pequena, a proporção real é calculada pela A5 em todas as máscaras).
- Inspeção visual de 4 pares imagem–máscara: máscaras alinhadas com os pólipos (incluindo o recorte retangular de uma imagem com janela preta). As imagens têm bordas pretas e texto sobreposto (datas).
- **Regra de binarização, igual para todo o grupo: máscara = píxel > 127.**

## 3. Unidade de agrupamento e divisão

- **Unidade de agrupamento:** a imagem (o Kvasir-SEG não tem identificador de doente).
- **Divisão:** 80/10/10 = 800/100/100 imagens, aleatória, repetida com 3 seeds (1, 2 e 3).
- **Ficheiros:** `data/splits_seed1.json`, `splits_seed2.json`, `splits_seed3.json`. Guardam **nomes de imagens**, não caminhos, e estão versionados no git.
- **Geração:** `src/dados.py`, função `criar_splits`, com `numpy.random.default_rng(seed)`. Gerados a 24/10/2026.
- **Verificações automáticas:** 1000 imagens, nomes de imagens e máscaras idênticos, ausência de sobreposição entre treino, validação e teste.
- **CVC-ClinicDB:** inteiro como teste externo; nunca usado para treinar, ajustar thresholds ou escolher modelos.

**Hashes SHA-256 dos ficheiros** (a divisão foi gerada duas vezes e os hashes coincidiram):

| Seed | SHA-256 |
|---|---|
| 1 | `2e59908720aa911efcba5568311b45fcc1743fac3df615b17b8a53226aea8cb7` |
| 2 | `8e54099dd6ab179c82c25b2e309c7016bf1b5d7edab517aac00ff06d6cf6a9e5` |
| 3 | `52feb9d9f8c522f53c211b3992e404f7fae2a22b5138148b3df01ce34d598b47` |

Antes de cada experiência, verificar que o hash do JSON usado coincide com esta tabela.

## 4. Limitação: ausência de identificador de doente

O Kvasir-SEG não indica quais as imagens que vêm do mesmo doente (ou do mesmo pólipo em ângulos diferentes). Imagens relacionadas podem cair em treino e em teste, o que gera **fuga de informação** e tende a **inflacionar** as métricas, sobretudo da U-Net, que aprende no domínio.

**Mitigações** (reduzem o problema, não o eliminam):
1. 3 seeds de divisão diferentes.
2. IC 95% por bootstrap de imagens.
3. Teste externo no CVC-ClinicDB, nunca usado para decisões.
4. Declaração da limitação no README, no relatório e na aplicação.

**Limitação residual:** o CVC-ClinicDB tem fotogramas correlacionados dentro de cada sequência.

## 5. Métrica principal e secundárias

- **Principal:** Dice médio por imagem (mDice) no teste, **na resolução original da máscara** (redimensiona-se a previsão, nunca a máscara real). É a métrica da literatura do Kvasir-SEG.
- **Secundárias:** mIoU; mDice por subgrupo; erro relativo da área, |A_prev − A_real| / A_real; desvio do Dice entre prompts com jitter; tempo por imagem.
- **Subgrupos:** pólipo pequeno (área < 2% da imagem), brilho especular (deteção por limiar de intensidade), bordos pouco nítidos. **[A5: fixar as definições operacionais]**
- **Binarização das máscaras:** `> 127` em todo o código (as máscaras JPEG não são estritamente 0/255).
- **Resolução:** com 333 resoluções distintas, qualquer redimensionamento (ex.: 352×352 na U-Net) deforma as imagens; a métrica calcula-se sempre na resolução original, redimensionando a **previsão** e nunca a máscara real.
- **Um único código** de avaliação: `src/avaliar.py` (A5), usado por todas as vias.

## 6. Condições a comparar

| Condição | Responsável |
|---|---|
| U-Net (3 seeds × 3 divisões) | A2 |
| SAM caixa oráculo | A3 |
| SAM ponto oráculo (centro e aleatório dentro do pólipo) | A3 |
| SAM caixa com jitter 10% (e 5, 20%) | A3 |
| SAM 2 caixa | A3 |
| SAM afinado (3 seeds × 3 divisões) | A4 |
| U-Net → caixa → SAM (automático) | A4 |

O prompt **oráculo** (derivado da máscara real) é um **limite superior** e identifica-se sempre como tal. Os cenários realistas são o jitter e o pipeline automático.

## 7. Baseline e método avançado

- **Baseline [A2]:** U-Net com encoder pré-treinado em ImageNet (ex.: ResNet-34), Dice+BCE, data augmentation, 352×352.
- **Data augmentation [A2: confirmar a lista final]:** só no treino; transformações geométricas aplicadas igualmente à imagem e à máscara; transformações de cor só na imagem; validação e teste sem augmentation.
- **Método avançado [A3, A4]:** SAM ViT-B e SAM 2 Hiera-S, zero-shot com prompt de ponto e caixa; depois afinação leve do descodificador de máscara (encoder congelado, embeddings pré-calculados, Dice+BCE).

## 8. Seeds e orçamento

- 3 seeds de treino por condição treinada.
- 5 prompts com jitter por imagem, com seed fixa.
- **Orçamento computacional:** a preencher com os tempos medidos na PL07 (Colab gratuito). **[todos]**

| Fase | N.º imagens | Épocas | Tempo/época medido | Total estimado | Cabe no Colab? |
|---|---|---|---|---|---|
| U-Net | 1000 | | | | |
| SAM afinado | 1000 | | | | |
| Avaliação (seeds, bootstrap) | 1000 | | | | |

## 9. Critério de seleção de modelo

Só com a **validação**: early stopping no mDice de validação; threshold da U-Net e escolha entre as 3 máscaras candidatas do SAM (a de maior score previsto, **nunca** a mais parecida com a máscara real) decididos na validação.

## 10. Análises previstas

- IC 95% por bootstrap de imagens (n = 1000).
- Bootstrap emparelhado da diferença U-Net vs SAM nas mesmas imagens.
- Análise por subgrupo.
- Curva Dice vs amplitude do jitter (0, 5, 10, 20%).
- 10 piores casos por condição.
- Queda Kvasir-SEG → CVC-ClinicDB.

## 11. Conjunto de teste e critério de sucesso

- **Uma única corrida por condição e por divisão**, depois do commit deste ficheiro. As máscaras previstas ficam guardadas em `runs/` (PNG) para recalcular qualquer métrica sem re-inferir.
- **Critério de sucesso:** H1 e H2 respondidas com IC. Mesmo que o SAM não supere a U-Net, a resposta quantificada à pergunta central (com robustez, subgrupos e teste externo) é o resultado.

**Verificações finais:**
- [ ] O Dice é calculado na resolução original da máscara.
- [ ] dice ≈ 2·iou / (1 + iou) em todas as imagens.
- [ ] Os valores da U-Net aproximam-se dos publicados para o Kvasir-SEG.
- [ ] O CVC-ClinicDB não entrou em nenhuma decisão.
- [ ] Todas as vias usam as mesmas divisões e o mesmo `src/avaliar.py`.

## 12. Alterações após o fecho

| Data | Alteração | Motivo | Quem |
|---|---|---|---|
| | | | |

---

*Citar Jha et al. (2020) e Bernal et al. (2015) sempre que os dados são mencionados.*
