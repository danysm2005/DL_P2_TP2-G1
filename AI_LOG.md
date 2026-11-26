# Registo do uso de IA (todos os membros)

Assistente usado: Claude (Anthropic). Cada linha diz o que a IA propôs, se estava correto e como foi verificado.

## Daniel Moreira — revisão do PROTOCOLO.md antes do fecho (21/11/2026)

| Proposta da IA (código ou afirmação) | Funcionou / estava correta? (S/N) | O que estava errado ou em falta | Como verificou (output, shape, métrica) |
|---|---|---|---|
| O desvio de 0.0000 entre seeds do SAM 2 afinado no CVC-ClinicDB indica que os pesos afinados não estavam a ser carregados | S | A função de carregamento dos pesos afinados não os aplicava corretamente | Depois da correção o desvio passou a 0.0039 (caixa) e 0.0038 (ponto), e o mDice mudou de 0.9204 para 0.9244 e de 0.7352 para 0.8719 |
| O protocolo descrevia a U-Net com Albumentations, Dice Loss e máx. 100 épocas | S (a descrição estava errada) | A U-Net é em Keras, com loss Dice+BCE, duas fases (≤25 e ≤80 épocas) e augmentation de flips, rot90, brilho e contraste | Leitura da célula de treino do notebook (`augment`, `dice_bce`, fases 1A e 1B) |
| O protocolo dizia «SAM 2 caixa: 28 épocas em 1h15» | S (a frase estava errada) | Os `info.yaml` dão entre 7 e 22 épocas por run; os tempos de treino não foram registados | `epocas_feitas` nos 12 `info.yaml` (7, 13, 8, 12, 12, 21, 9, 16, 8, 18, 8, 22) |
| O protocolo dizia que o SAM afinado escolhe a máscara por `argmax` do score | S (a frase estava errada) | O `argmax` é só do zero-shot; o afinado usa uma única máscara (`multimask_output=False`) | `predict(...)` em `2.py` e `3.py` |

## Pedidos obrigatórios (completar)

| Semana | Pedido | Resultado e verificação |
|---|---|---|
| 8 (A4) | Loop de afinação do descodificador do SAM com embeddings pré-calculados e Dice+BCE | a preencher: encoder sem gradientes, loss a descer, Dice de validação acima do zero-shot |
| 9 (A1) | Identificar fugas de dados, métricas inadequadas e comparações injustas no PROTOCOLO.md | ver anexo C do `PROTOCOLO.md` |
| 14 (todos) | Marcar cada frase da discussão como suportada ou não pelos números | a preencher |

## Outros membros

*(Eliana Monteiro, Sara Costa, José Peixoto: acrescentar as vossas entradas.)*
