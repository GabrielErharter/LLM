# Resultados dos Experimentos de Treinamento

- Texto: `texto.txt` (400,944 caracteres)
- Configuracao base: `{"tokenizador": "char", "contexto": 64, "batch": 16, "emb": 128, "heads": 4, "camadas": 4, "dropout": 0.1, "lr": 0.0005, "wd": 0.1, "epocas": 10, "split": 0.9, "seed": 123, "stride": null, "eval_freq": 20, "eval_iter": 5, "grad_clip": null}`
- Baseline (chute uniforme): perplexity = tamanho do vocabulario; loss = ln(V)
- Cada experimento altera apenas o parametro indicado.

======================================================================
## Experimento D - Numero de epocas (curva do treino base)
======================================================================
Baseline ln(112) = 4.7185; loss inicial: treino=4.8692, val=4.7966

| epoca | loss treino | loss val | ppl val | val-treino |
|---|---|---|---|---|
| 1 | 2.3545 | 3.3897 | 29.7 | +1.0352 |
| 2 | 2.1467 | 3.2161 | 24.9 | +1.0694 |
| 3 | 2.0066 | 3.0614 | 21.4 | +1.0548 |
| 4 | 1.9099 | 2.9096 | 18.3 | +0.9997 |
| 5 | 1.8358 | 2.8509 | 17.3 | +1.0151 |
| 6 | 1.7661 | 2.7835 | 16.2 | +1.0174 |
| 7 | 1.7136 | 2.7132 | 15.1 | +0.9996 |
| 8 | 1.6676 | 2.7451 | 15.6 | +1.0774 |
| 9 | 1.6315 | 2.6920 | 14.8 | +1.0605 |
| 10 | 1.5986 | 2.7069 | 15.0 | +1.1083 |

Melhor epoca pela loss de validacao: 9. Sinal de sobreajuste: nao.

Amostras (greedy, mesmo prompt) em epocas selecionadas:

**Epoca 1**
```
The Project Gutenberg eBook of Dom Casmu de de de de de de e de de de e e e e e e de e e e de e e e e e e e de e e de de de de de e e e e e e de de e e e e e e e e e de de de e e e e de e e 
```

**Epoca 5**
```
The Project Gutenberg eBook of Dom Casmuito de para de como de para de para de para de para de para de para de para de para de para de para de para de para de para de para de para de para de
```

**Epoca 10**
```
The Project Gutenberg eBook of Dom Casmura a para de minha mãe
achara de minha mãe de contar a minha mãe de contar a minha mãe de
contar e a minha mãe de contar a minha mãe de contar a minha
```

======================================================================
## Experimento A - Taxa de aprendizado
======================================================================
| lr | loss treino | loss val | ppl val | val-treino | melhor epoca (val) | sobreajuste? | tempo (s) | tokens/s |
|---|---|---|---|---|---|---|---|---|
| lr=0.0001 | 2.0903 | 3.2170 | 25.0 | +1.1267 | 10 | nao | 297.9 | 12100 |
| lr=0.0005 | 1.5986 | 2.7069 | 15.0 | +1.1083 | 9 | nao | 268.8 | 13410 |
| lr=0.002 | 1.4340 | 2.5626 | 13.0 | +1.1285 | 9 | SIM | 263.6 | 13672 |
| lr=0.01 | 1.6880 | 2.6498 | 14.2 | +0.9618 | 10 | nao | 274.6 | 13127 |

Amostras ao final do treino (greedy, prompt `The Project Gutenberg eBook of Dom Casmu`):

**lr=0.0001**
```
The Project Gutenberg eBook of Dom Casmuito de a de por de a de a de parar a de a de a de parara a a de a de a de parara a a de a de a de parara a a de a de a de parara a a de a de a de para
```

**lr=0.0005**
```
The Project Gutenberg eBook of Dom Casmura a para de minha mãe
achara de minha mãe de contar a minha mãe de contar a minha mãe de
contar e a minha mãe de contar a minha mãe de contar a minha
```

**lr=0.002**
```
The Project Gutenberg eBook of Dom Casmuro de contrar a minha mãe de contrar a minha mãe de contrar a minha mãe de contrar a minha mãe de contrar a minha mãe de contrar a minha mãe de contra
```

**lr=0.01**
```
The Project Gutenberg eBook of Dom Casmuro de com a minha mãe a minha a minha a minha a minha a minha a minha a minha a minha a minha a minha a minha a minha a minha a minha a minha a minha 
```

======================================================================
## Experimento B - Tamanho do lote
======================================================================
| batch | loss treino | loss val | ppl val | val-treino | melhor epoca (val) | sobreajuste? | tempo (s) | tokens/s |
|---|---|---|---|---|---|---|---|---|
| batch=4 | 1.5046 | 2.5796 | 13.2 | +1.0750 | 10 | nao | 483.0 | 7468 |
| batch=16 | 1.5986 | 2.7069 | 15.0 | +1.1083 | 9 | nao | 271.6 | 13270 |
| batch=64 | 1.8443 | 2.8560 | 17.4 | +1.0117 | 10 | nao | 286.9 | 12563 |

Amostras ao final do treino (greedy, prompt `The Project Gutenberg eBook of Dom Casmu`):

**batch=4**
```
The Project Gutenberg eBook of Dom Casmulo de Capitú de Capitú e a minha mãe era mãe a minha mãe era mãe a minha mãe era mãe a minha mãe era mãe a minha mãe era mãe a minha mãe era mãe a min
```

**batch=16**
```
The Project Gutenberg eBook of Dom Casmura a para de minha mãe
achara de minha mãe de contar a minha mãe de contar a minha mãe de
contar e a minha mãe de contar a minha mãe de contar a minha
```

**batch=64**
```
The Project Gutenberg eBook of Dom Casmuito de sem a mais de mas paras de mas de mas paras de mas de mas paras de a mais de mestas de mestas de mestas de mestas de mestas de mestas de mestas
```

======================================================================
## Experimento C - Dropout
======================================================================
| dropout | loss treino | loss val | ppl val | val-treino | melhor epoca (val) | sobreajuste? | tempo (s) | tokens/s |
|---|---|---|---|---|---|---|---|---|
| dropout=0 | 1.4380 | 2.6672 | 14.4 | +1.2291 | 10 | nao | 206.1 | 17489 |
| dropout=0.1 | 1.5986 | 2.7069 | 15.0 | +1.1083 | 9 | nao | 275.5 | 13082 |
| dropout=0.3 | 1.8387 | 2.8276 | 16.9 | +0.9889 | 9 | SIM | 263.5 | 13680 |

Amostras ao final do treino (greedy, prompt `The Project Gutenberg eBook of Dom Casmu`):

**dropout=0**
```
The Project Gutenberg eBook of Dom Casmulho primeiro a minha mãe
de casar a minha mãe a minha mãe a minha mãe a minha mãe era compliciosico. Esta meia em complicio de mesmo que elle para mes
```

**dropout=0.1**
```
The Project Gutenberg eBook of Dom Casmura a para de minha mãe
achara de minha mãe de contar a minha mãe de contar a minha mãe de
contar e a minha mãe de contar a minha mãe de contar a minha
```

**dropout=0.3**
```
The Project Gutenberg eBook of Dom Casmuitú de a minha de a minha de a minha de a minha de a minha de a minha de a minha de a minha de a minha de a minha de a minha de a minha de a minha de 
```
