# PROTOCOLO.md: protocolo experimental

Tema A · Segmentação de pólipos em colonoscopia (Kvasir-SEG) · Grupo `DL_P2_TP1-G1`

**Estado:** Protocolo escrito na aula PL09 (12/11/2026). A secção 7 (resultados) foi acrescentada depois, quando o conjunto de teste já tinha sido avaliado. As secções 1 a 6 só foram corrigidas onde descreviam o código de forma diferente da realmente usada (*augmentation*, *loss*, fases de treino da U-Net, tempos de treino); nenhum critério nem análise prevista foi alterado.

---

## 1. Questão central e hipóteses

**Questão central:** até que ponto os modelos fundacionais SAM e SAM 2,
orientados por prompts e sem treino específico inicial no domínio da
colonoscopia, conseguem aproximar-se ou superar uma U-Net treinada com
imagens do Kvasir-SEG? Como varia o desempenho com o tipo de prompt e
com a imprecisão das indicações?

| Id | Hipótese |
|---|---|
| H1 | O SAM zero-shot com caixa oráculo atinge um mDice não inferior ao da U-Net (IC 95% da diferença inclui 0 ou é favorável ao SAM). |
| H2 | A afinação leve do descodificador de máscara supera o zero-shot e a U-Net, com IC da diferença 95% a excluir 0. |
| H3 | O prompt de caixa apresenta melhor desempenho do que o ponto e perde menos de 3 pontos de Dice com jitter de 10%. |
| H4 | O desempenho diminui em casos difíceis: pólipos pequenos (área inferior a 2% da imagem), nas imagens com brilho especular e em imagens com limites pouco definidos. |
| H5 | No CVC-ClinicDB a U-Net degrada-se mais do que o SAM zero-shot devido à mudança de distribuição. |








## 2. Dados

| Conjunto | Papel | Dimensão |
|---|---|---|
| Kvasir-SEG (Jha et al., 2020) | treino, validação, teste | 1000 pares imagem–máscara |
| CVC-ClinicDB (Bernal et al., 2015) | teste externo | 612 fotogramas, 31 sequências |

###2.2. Verificação dos dados
Verificação realizada em 24/10/2026 pelo elemento A1.
- 1000 imagens e 1000 máscaras do Kvasir-SEG, com nomes coincidentes; 0 pares com imagem e máscara de tamanhos diferentes.
- Largura entre 332 e 1920 e altura entre 352 e 1072, com 333 resoluções distintas. A resolução mais frequente foi 622×530 (78 imagens). Tabela completa em `resolucoes.csv`.
- Em 10 máscaras aleatórias (seed 0): valores de 0 a 255, com 13 a 16 níveis distintos por artefactos de compressão JPEG (as máscaras são `.jpg`) e 100% dos píxeis a menos de 10 níveis de 0 ou 255.
- Área do pólipo nessas 10 máscaras: entre 1,2% e 50,7% da imagem (só 1 em 10 abaixo de 2%; é uma amostra pequena, a proporção real é calculada pela A5 em todas as máscaras).
- Inspeção visual de 4 pares imagem–máscara: máscaras alinhadas com os pólipos (incluindo o recorte retangular de uma imagem com janela preta). As imagens têm bordas pretas e texto sobreposto (datas).
- **Regra de binarização:** máscara binária obtida por
`pixel > 127`, aplicada de forma consistente em todo o código.

### 2.2. Papel do CVC-ClinicDB

O CVC-ClinicDB é mantido integralmente como conjunto de teste externo.
Não é utilizado no treino, na seleção de modelos, na escolha de limiares
(*thresholds*), na afinação de hiperparâmetros ou em qualquer outra
decisão metodológica. A sua utilização destina-se exclusivamente a
avaliar a generalização para dados de outra origem.

Os fotogramas do CVC-ClinicDB apresentam correlação dentro das
respetivas sequências de vídeo; esta característica é reconhecida como
limitação na interpretação das métricas agregadas.

## 3. Unidade de agrupamento e divisão

- **Unidade de agrupamento:** a imagem. O Kvasir-SEG não disponibiliza  identificadores de doente que permitam uma divisão por paciente.
- **Divisão:** 80/10/10, correspondendo a 800 imagens de treino, 100 de validação e 100 de teste.
- **Repetições:** três divisões independentes, associadas às seeds 1,
    2 e 3.
- **Ficheiros:** `data/splits_seed1.json`, `splits_seed2.json`, `splits_seed3.json`. Guardam nomes de imagens, não caminhos, e estão versionados no git.
- **Geração:** função `criar_splits` em `src/dados.py`, utilizando
    `numpy.random.default_rng(seed)`. Gerados em 24/10/2026.
- **Verificações:** 1000 imagens; correspondência entre nomes de
    imagens e máscaras; ausência de sobreposição entre treino, validação
    e teste.

### 3.1. Integridade das divisões --- SHA-256

Os ficheiros foram gerados duas vezes e os hashes coincidiram. Antes de
cada experiência, deve confirmar-se que o hash do JSON utilizado
corresponde ao hash registado.


| Ficheiro |Treino| Validação | Teste | SHA-256 |
|---|---|---|---|---|
| `data/splits_seed1.json` | 800 | 100 | 100 |`2e59908720aa911efcba5568311b45fcc1743fac3df615b17b8a53226aea8cb7` |
|  `data/splits_seed2.json` | 800 | 100 | 100 | `8e54099dd6ab179c82c25b2e309c7016bf1b5d7edab517aac00ff06d6cf6a9e5` |
| `data/splits_seed3.json` | 800 | 100 | 100 | `52feb9d9f8c522f53c211b3992e404f7fae2a22b5138148b3df01ce34d598b47` |

## 4. Limitações da divisão e risco de fuga de informação

O Kvasir-SEG não inclui identificadores de doente. Assim, imagens
relacionadas --- por exemplo, imagens do mesmo exame ou do mesmo pólipo
em ângulos diferentes --- podem ficar em partições distintas. Esta
situação pode provocar fuga de informação e inflacionar as métricas,
sobretudo em modelos treinados no próprio domínio.

Medidas de mitigação, que reduzem o risco mas não o eliminam:

1.  Três divisões com seeds diferentes.
2.  Intervalos de confiança por *bootstrap* ao nível da imagem.
3.  Avaliação externa no CVC-ClinicDB, sem utilização desse conjunto
    para decisões.
4.  Declaração explícita desta limitação no README e no relatório.

**Limitação residual:** o CVC-ClinicDB contém fotogramas correlacionados
dentro de cada sequência; as imagens desse conjunto também não devem ser
interpretadas como observações totalmente independentes.

## 5. Métricas

### 5.1. Métrica principal

-   **Mean Dice (mDice):** média do Dice por imagem.
-   A avaliação é feita no teste, na resolução original da máscara.
-   Quando necessário, redimensiona-se a previsão para a resolução
    original; a máscara de referência não é redimensionada.
-   São apresentados intervalos de confiança de 95% por *bootstrap*
    (2000 reamostragens), conforme o procedimento registado na PL09.

### 5.2. Métricas secundárias

-   **Mean IoU (mIoU):** interseção sobre a união.
-   **Erro relativo da área:** `|A_prevista − A_real| / A_real`.
-   **Taxa de falhas graves:** percentagem de imagens com Dice inferior
    a 0,5.
-   **Estabilidade dos prompts:** variação do Dice com diferentes
    amplitudes de *jitter* e diferentes pontos.
-   **Tempo médio de inferência por imagem.**

### 5.3. Subgrupos

-   Pólipos pequenos: área inferior a 2% da imagem.
-   Imagens com reflexos especulares.
-   Imagens com limites pouco definidos.

As definições operacionais utilizadas no código encontram-se em
`src/avaliar.py`, incluindo `brilho_especular` e `nitidez_bordo`.

### 5.4. Consistência da avaliação

-   Binarização das máscaras com `pixel > 127`.
-   Todas as condições são avaliadas com o mesmo código:
    `src/avaliar.py`.
-   Avaliação na resolução original da máscara, redimensionando apenas a
    previsão.
-   Como verificação, deve confirmar-se a relação aproximada
    `Dice = 2 × IoU / (1 + IoU)` por imagem.

## 6. Condições Experimentais a comparar

As condições são avaliadas nas mesmas divisões do Kvasir-SEG.

| Condição | Prompt/Configuração | Responsável |
|---|---|
| U-Net (3 seeds × 3 divisões) | Baseline Supervisionada| A5 |
| SAM zero-shot | Caixa oráculo | A3 |
| SAM zero-shot | Ponto central e ponto aleatório dentro do pólipo | A3 |
| SAM 2 zero-shot | Caixa oráculo | A3 |
| SAM 2 zero-shot | Ponto central e ponto aleatório dentro do pólipo | A3 |
| SAM com afinação do descodificador | Caixa e ponto | A3 |
| SAM 2 com afinação do descodificador | Caixa e ponto | A3 |
| Robustez a *jitter* | Caixas perturbadas em 5%, 10% e 20%; cinco repetições por imagem | A3 |
| Robustez a pontos | Cinco cliques aleatórios dentro do pólipo | A3 |
| Pipeline automático | U-Net → caixa delimitadora → SAM  (integração da aplicação) | A4 |

### 6.1. Interpretação dos prompts

Os prompts oráculo (ponto ou caixa derivados da máscara real) constituem
um limite superior, pois dependem de informação da anotação de
referência que não estaria disponível dessa forma numa utilização real.
Devem ser identificados explicitamente como oráculo.

Os cenários que simulam indicações menos precisas são os prompts com
*jitter* e os pontos aleatórios. O pipeline automático U-Net → caixa →
SAM só deve ser apresentado como resultado experimental se tiver sido
efetivamente executado e avaliado.

## 7. Modelos, treino e implementação

### 7.1. Baseline --- U-Net (A5)

-   U-Net com *encoder* ResNet-34 pré-treinado em ImageNet.
-   Resolução de entrada: 352 × 352 píxeis.
-   Função de perda: combinação Dice + Binary Cross-Entropy (BCE).
-   *Data augmentation* apenas no conjunto de treino.
-   Transformações geométricas aplicadas igualmente à imagem e à
    máscara; transformações de cor aplicadas apenas à imagem.
-   Configuração de *augmentation* registada: espelhamentos horizontal e
    vertical, rotações de 90°, variação de brilho (±0,1) e contraste
    (0,8--1,2).
-   Treino em duas fases, com *early stopping* baseado em `val_mdice`:
    -   **Fase 1A:** *encoder* congelado, `lr = 1e-3`, máximo de 25
        épocas.
    -   **Fase 1B:** *encoder* descongelado, `lr = 1e-4`, máximo de 80
        épocas.
-   Épocas executadas nas seeds 1, 2 e 3: 82, 71 e 53, respetivamente.

### 7.2. Modelos fundacionais --- SAM 1 e SAM 2 (A3)

-   **SAM 1:** encoder ViT-B (`vit_b`).
-   **SAM 2:** encoder Hiera-S (`hiera_s`).
-   Ambos são avaliados inicialmente em modo *zero-shot*, sem treino
    específico no Kvasir-SEG.
-   São utilizados prompts de caixa e de ponto.
-   É também avaliada a afinação leve do descodificador de máscara,
    mantendo o encoder congelado e utilizando embeddings pré-calculados.
-   Função de perda no *fine-tuning*: Dice + BCE.
-   Configuração de treino registada: máximo de 30 épocas, *early
    stopping* com paciência 5, `lr = 5e-5`, `weight_decay = 0.01`,
    `batch_size = 8` e `jitter_treino = 0.1`.
-   Foram executadas entre 7 e 22 épocas por *run*.
-   Na inferência zero-shot, quando são produzidas três máscaras
    candidatas, seleciona-se a máscara com maior *predicted IoU score*,
    e não a mais semelhante à máscara real.
-   No *fine-tuning*, foi utilizada uma única máscara por prompt
    (`multimask_output=False`).

## 8. Seeds e Orçamento Computacional

-   Três seeds (1, 2 e 3) nas condições treinadas.
-   Cinco repetições por imagem para cada nível de *jitter*.
-   Execução no Google Colab gratuito, com GPU T4.

| Fase | N.º imagens | Épocas | Tempo/época medido | Total estimado | Cabe no Colab? |
|---|---|---|---|---|---|
| U-Net | 1000 | 100 (x3 seeds) | ~30s/época | ~25 min (por seed) | Sim |
| SAM afinado | 1000 | 30 (x3 seeds) | ~2min/época | ~60min (por modelo) | Sim |
| Avaliação (seeds, bootstrap) | 1000 | 1 passagem | ~1min/split | ~15min (por bootstrap) | Sim |

## 9. Critério de seleção de modelo

A seleção é realizada exclusivamente com os dados de validação:

-   O *checkpoint* final de cada treino corresponde à época com maior
    `val_mdice` na validação.
-   O limiar de binarização da U-Net é selecionado na validação, dentro
    do intervalo testado de 0,3 a 0,7.
-   No SAM/SAM 2 zero-shot, a seleção entre as três máscaras candidatas
    é feita pelo maior *predicted IoU score* do próprio modelo, nunca
    pela comparação com a máscara de referência.
-   O teste do Kvasir-SEG e o CVC-ClinicDB não são utilizados para
    escolher modelos, limiares ou hiperparâmetros.

## 10. Análises previstas

1.  Intervalos de confiança de 95% por *bootstrap* (2000 reamostragens).
2.  Comparações emparelhadas da diferença de mDice entre cada condição e
    a U-Net, nas mesmas imagens.
3.  Análise por subgrupos: pólipos pequenos, reflexos especulares e
    limites pouco definidos.
4.  Curva de desempenho em função do *jitter* das caixas (0%, 5%, 10% e
    20%).
5.  Comparação entre ponto central e pontos aleatórios dentro do pólipo.
6.  Relação entre estimativas de qualidade/incerteza dos modelos e o
    Dice observado.
7.  Inspeção visual dos 10 piores casos por condição.
8.  Comparação da generalização entre Kvasir-SEG e CVC-ClinicDB.
9.  Registo do tempo de inferência por imagem.

## 11. Avaliação no teste e critério de sucesso

### 11.1. Conjunto de teste

-   Cada condição é avaliada uma única vez por divisão, depois de
    congelado e versionado o protocolo.
-   As máscaras previstas são guardadas em `runs/` para permitir
    recalcular métricas sem repetir a inferência.
-   O CVC-ClinicDB é avaliado como conjunto externo, sem treino,
    afinação, seleção ou ajuste de limiares nesse conjunto.
-   Qualquer correção de erro de implementação que obrigue a repetir uma
    avaliação deve ser documentada na secção 13, identificando o erro, a
    correção e as condições reavaliadas.

### 11.2. Critério de sucesso

Considera-se evidência de melhoria relevante quando uma condição SAM/SAM
2 apresenta:

-   aumento de pelo menos 0,0200 no mDice face à U-Net
    (`ΔmDice ≥ +0.0200`); e
-   intervalo de confiança de 95% da diferença emparelhada inteiramente
    acima de zero.

A interpretação final deve também considerar a robustez aos prompts e o
desempenho no CVC-ClinicDB. O não cumprimento do critério não invalida o
estudo: a resposta quantitativa à pergunta central, incluindo
limitações, variabilidade e falhas, constitui o resultado do projeto.

### 11.3. Verificações finais

-   [ ] O Dice é calculado na resolução original da máscara.
-   [ ] A relação aproximada `Dice = 2 × IoU / (1 + IoU)` é verificada
    por imagem.
-   [ ] Todas as condições usam as mesmas divisões e o mesmo
    `src/avaliar.py`.
-   [ ] Os hashes das divisões correspondem aos valores registados.
-   [ ] O CVC-ClinicDB não foi utilizado em qualquer decisão
    metodológica.
-   [ ] Os prompts oráculo estão identificados como tal.
-   [ ] Os resultados de *jitter* e dos subgrupos correspondem às
    configurações efetivamente executadas.

## 12. Resultados registados

### 12.1. Kvasir-SEG — conjunto de teste ($100$ imagens $\times$ $3$ *seeds*)

| Modelo | mDice (Média ± DP) | IC 95% *bootstrap* (mDice) | mIoU (Média ± DP) | Erro Rel. Área (%) | Falhas Dice<0.5 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| SAM2 FT (Caixa) | 0.9372 ± 0.0038 | [0.9258, 0.9475] | 0.8928 ± 0.0048 | 8.93 ± 1.97 | 2.3 ± 0.6 |
| SAM1 FT (Caixa) | 0.9231 ± 0.0021 | [0.9109, 0.9336] | 0.8695 ± 0.0016 | 10.17 ± 2.11 | 1.3 ± 1.2 |
| SAM2 FT (Ponto) | 0.9175 ± 0.0159 | [0.9054, 0.9285] | 0.8598 ± 0.0208 | 14.16 ± 5.32 | 1.7 ± 2.1 |
| U-Net | 0.8995 ± 0.0145 | [0.8817, 0.9155] | 0.8402 ± 0.0162 | 19.13 ± 0.64 | 3.7 ± 1.5 |
| SAM1 FT (Ponto) | 0.8918 ± 0.0123 | [0.8765, 0.9058] | 0.8233 ± 0.0119 | 19.50 ± 7.76 | 2.7 ± 3.1 |
| SAM2 ZS (Caixa) | 0.8795 ± 0.0180 | [0.8569, 0.9011] | 0.8215 ± 0.0158 | 23.99 ± 5.27 | 6.0 ± 3.6 |
| SAM1 ZS (Caixa) | 0.8624 ± 0.0124 | [0.8391, 0.8848] | 0.7963 ± 0.0125 | 31.60 ± 6.91 | 6.7 ± 1.2 |
| SAM2 ZS (Ponto) | 0.7466 ± 0.0411 | [0.7098, 0.7819] | 0.6685 ± 0.0395 | 88.28 ± 64.53 | 19.0 ± 6.2 |
| SAM1 ZS (Ponto) | 0.5288 ± 0.0409 | [0.4881, 0.5691] | 0.4448 ± 0.0407 | 285.46 ± 51.61 | 49.3 ± 5.5 |

Diferenças emparelhadas contra a U-Net (*bootstrap*, IC 95% da diferença):

| Comparação | Δ mDice | IC 95% (Δ) | Significativo |
| :--- | :---: | :---: | :---: |
| SAM2 FT (Caixa) vs. U-Net | +0.0377 | [+0.0218, +0.0537] | Sim |
| SAM1 FT (Caixa) vs. U-Net | +0.0236 | [+0.0074, +0.0401] | Sim |
| SAM2 FT (Ponto) vs. U-Net | +0.0179 | [+0.0030, +0.0333] | Sim |
| SAM1 FT (Ponto) vs. U-Net | -0.0078 | [-0.0251, +0.0099] | Não |
| SAM2 ZS (Caixa) vs. U-Net | -0.0200 | [-0.0464, +0.0051] | Não |
| SAM1 ZS (Caixa) vs. U-Net | -0.0372 | [-0.0646, -0.0110] | Sim |
| SAM2 ZS (Ponto) vs. U-Net | -0.1529 | [-0.1861, -0.1201] | Sim |
| SAM1 ZS (Ponto) vs. U-Net | -0.3707 | [-0.4108, -0.3297] | Sim |

### 12.2. Generalização externa — CVC-ClinicDB ($612$ imagens)

Avaliação sem qualquer afinação no CVC-ClinicDB. Os modelos treinados (U-Net e SAM/SAM 2 FT) usam os pesos de cada uma das $3$ *seeds*; a U-Net usa o limiar escolhido na validação de cada *seed* ($0.3$, $0.7$ e $0.5$). Os modelos *zero-shot* têm uma única corrida (sem treino), por isso o desvio é $0.0000$. Δ mDice = mDice no CVC − mDice no Kvasir-SEG (secção 7.1). A primeira execução do CVC tinha dois erros de código (os pesos *fine-tuned* do SAM 2 não eram carregados corretamente e a U-Net usava limiar fixo $0.5$ em vez do limiar da validação); foi repetida com o código corrigido, sem alterar modelos, hiperparâmetros nem critérios. Os valores abaixo são os da execução corrigida.

| Modelo | CVC mDice (Média ± DP) | CVC mIoU (Média ± DP) | Erro Rel. Área (%) | Falhas Dice<0.5 (%) | Δ mDice (CVC − Kvasir) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| SAM2 FT (Caixa) | 0.9244 ± 0.0039 | 0.8694 ± 0.0045 | 9.82% | 0.8% | -0.0128 |
| SAM2 ZS (Caixa) | 0.9211 ± 0.0000 | 0.8661 ± 0.0000 | 16.96% | 1.3% | +0.0416 |
| SAM1 FT (Caixa) | 0.9103 ± 0.0057 | 0.8505 ± 0.0084 | 18.70% | 1.7% | -0.0128 |
| SAM1 ZS (Caixa) | 0.8968 ± 0.0000 | 0.8326 ± 0.0000 | 21.60% | 2.1% | +0.0344 |
| SAM2 FT (Ponto) | 0.8719 ± 0.0038 | 0.7965 ± 0.0038 | 45.95% | 3.8% | -0.0456 |
| U-Net | 0.8361 ± 0.0101 | 0.7597 ± 0.0141 | 26.44% | 8.1% | -0.0634 |
| SAM1 FT (Ponto) | 0.7973 ± 0.0115 | 0.7050 ± 0.0105 | 102.15% | 10.7% | -0.0945 |
| SAM2 ZS (Ponto) | 0.7361 ± 0.0000 | 0.6590 ± 0.0000 | 217.44% | 22.1% | -0.0105 |
| SAM1 ZS (Ponto) | 0.4467 ± 0.0000 | 0.3691 ± 0.0000 | 760.76% | 63.6% | -0.0821 |

U-Net por *seed* no CVC: $0.8390$ (*seed* 1), $0.8444$ (*seed* 2), $0.8249$ (*seed* 3).

### 12.3. Estabilidade perante *jitter* na caixa (Kvasir-SEG, teste, $3$ *seeds*)

mDice (Média ± DP) com $5$ repetições por imagem para $5\%$, $10\%$ e $20\%$; $0\%$ é a caixa exata. Para comparação, a U-Net (sem *prompt*) tem $0.8995$.

| Modelo | Jitter 0% | Jitter 5% | Jitter 10% | Jitter 20% |
| :--- | :---: | :---: | :---: | :---: |
| SAM 1 ZS (Caixa) | 0.8624 ± 0.0124 | 0.8678 ± 0.0070 | 0.8575 ± 0.0059 | 0.8114 ± 0.0143 |
| SAM 2 ZS (Caixa) | 0.8795 ± 0.0180 | 0.8721 ± 0.0180 | 0.8567 ± 0.0144 | 0.7969 ± 0.0147 |
| SAM 1 FT (Caixa) | 0.9231 ± 0.0021 | 0.9202 ± 0.0022 | 0.9109 ± 0.0017 | 0.8680 ± 0.0044 |
| SAM 2 FT (Caixa) | 0.9372 ± 0.0038 | 0.9370 ± 0.0019 | 0.9285 ± 0.0021 | 0.8839 ± 0.0036 |

Ponto central vs. ponto aleatório ($5$ cliques dentro do pólipo):

| Modelo | Ponto central | Ponto aleatório |
| :--- | :---: | :---: |
| SAM 1 ZS | 0.5288 ± 0.0409 | 0.5352 ± 0.0284 |
| SAM 2 ZS | 0.7466 ± 0.0411 | 0.7313 ± 0.0449 |
| SAM 1 FT | 0.8918 ± 0.0123 | 0.7855 ± 0.0365 |
| SAM 2 FT | 0.9175 ± 0.0159 | 0.8332 ± 0.0176 |

### 12.4. Critério de sucesso

O critério (secção 1) exige $\Delta\text{mDice} \ge +0.0200$ face à U-Net, com IC 95% da diferença estritamente acima de zero, preservando a robustez sob *jitter* e na generalização externa.

* **$\Delta\text{mDice}$ no Kvasir-SEG:** cumprido por SAM 2 FT (Caixa) ($+0.0377$) e SAM 1 FT (Caixa) ($+0.0236$). SAM 2 FT (Ponto) tem IC acima de zero mas $+0.0179$, abaixo do limiar de $+0.0200$. SAM 1 FT (Ponto) não difere da U-Net.
* ***Jitter*:** com $5\%$ e $10\%$ de *jitter*, SAM 1 FT e SAM 2 FT (Caixa) continuam acima da U-Net ($0.8995$); com $20\%$ ficam abaixo ($0.8680$ e $0.8839$).
* **CVC-ClinicDB:** SAM 2 FT (Caixa) ($0.9244$) e SAM 1 FT (Caixa) ($0.9103$) ficam acima da U-Net ($0.8361$). A diferença entre SAM 2 FT (Caixa) e SAM 2 ZS (Caixa) no CVC é de $0.0033$, e o ZS tem uma só corrida, pelo que não foi testada.
* **Nota sobre os *prompts*:** a caixa e o ponto central são derivados da máscara real (oráculo) e constituem um limite superior; os cenários realistas são o *jitter* (secção 7.3) e o *pipeline* automático U-Net → caixa → SAM, que **não** foi avaliado neste protocolo.

### 12.5. Subgrupos clínicos (Kvasir-SEG, teste, $3$ *seeds*)

Subgrupos definidos em `src/avaliar.py`: pólipos pequenos (área $< 2\%$ da imagem), reflexos especulares (`brilho_especular`) e limites pouco definidos (`nitidez_bordo`, gradiente médio no contorno).

| Modelo | Geral (mDice) | Pólipos pequenos | Com reflexos | Limites pouco definidos |
| :--- | :---: | :---: | :---: | :---: |
| SAM2 FT (Caixa) | 0.9372 | 0.9429 | 0.9359 | 0.9409 |
| SAM1 FT (Caixa) | 0.9231 | 0.9421 | 0.9257 | 0.9273 |
| SAM2 FT (Ponto) | 0.9175 | 0.8620 | 0.9182 | 0.9114 |
| U-Net | 0.8995 | 0.6538 | 0.8955 | 0.9032 |
| SAM1 FT (Ponto) | 0.8918 | 0.8474 | 0.8817 | 0.8920 |
| SAM2 ZS (Caixa) | 0.8795 | 0.9331 | 0.8627 | 0.9125 |
| SAM1 ZS (Caixa) | 0.8624 | 0.9299 | 0.8436 | 0.9055 |
| SAM2 ZS (Ponto) | 0.7466 | 0.4967 | 0.7252 | 0.7282 |
| SAM1 ZS (Ponto) | 0.5288 | 0.4695 | 0.5286 | 0.4351 |

### 12.6. Tempo de inferência

Medido em `tempos.json`: SAM 1 (`vit_b`), caixa e ponto, $0.87$ s/imagem; U-Net, $0.0956$ s/imagem. SAM 2 e variantes *fine-tuned*: não registado.


## 13. Registo de Alterações

| Data | Alteração | Motivo | Responsável |
|---|---|---|---|
| | | | |

---


## Referências dos conjuntos de dados

-   Jha et al. (2020) --- Kvasir-SEG.
-   Bernal et al. (2015) --- CVC-ClinicDB.

Citar estas referências sempre que os conjuntos de dados forem descritos
no relatório, README ou aplicação.
