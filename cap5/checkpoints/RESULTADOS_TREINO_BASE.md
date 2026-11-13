# Resultados do treinamento base

- Texto: `texto.txt` (400,944 caracteres)
- Dados: {'n_tokens': 400944, 'n_treino': 360849, 'n_validacao': 40095, 'lotes_treino': 352, 'lotes_validacao': 40, 'vocab': 112}
- Parametros do treino: `{"tokenizador": "char", "contexto": 64, "batch": 16, "emb": 128, "heads": 4, "camadas": 4, "dropout": 0.1, "lr": 0.0005, "wd": 0.1, "epocas": 10, "split": 0.9, "seed": 123, "stride": null, "eval_freq": 20, "eval_iter": 5, "grad_clip": null}`
- Parametros do modelo: 828,672
- Dispositivo: cpu
- Tempo total: 297.7s

Baseline: ln(112) = 4.7185 (perplexity 112).

| epoca | loss treino | loss val | ppl treino | ppl val | tempo (s) |
|---|---|---|---|---|---|
| 0 (inicial) | 4.8692 | 4.7966 | 130.2 | 121.1 | - |
| 1 | 2.3545 | 3.3897 | 10.5 | 29.7 | 25.3 |
| 2 | 2.1467 | 3.2161 | 8.6 | 24.9 | 24.9 |
| 3 | 2.0066 | 3.0614 | 7.4 | 21.4 | 28.9 |
| 4 | 1.9099 | 2.9096 | 6.8 | 18.3 | 30.2 |
| 5 | 1.8358 | 2.8509 | 6.3 | 17.3 | 39.6 |
| 6 | 1.7661 | 2.7835 | 5.8 | 16.2 | 32.9 |
| 7 | 1.7136 | 2.7132 | 5.5 | 15.1 | 29.2 |
| 8 | 1.6676 | 2.7451 | 5.3 | 15.6 | 29.2 |
| 9 | 1.6315 | 2.6920 | 5.1 | 14.8 | 28.7 |
| 10 | 1.5986 | 2.7069 | 4.9 | 15.0 | 28.9 |

## Amostras (greedy, mesmo prompt) por epoca

Prompt: `The Project Gutenberg eBook of Dom Casmu`

**Epoca 1**
```
The Project Gutenberg eBook of Dom Casmu de de de de de de e de de de e e e e e e de e e e de e e e e e e e de e e de de de de de e e e e e e de de e e e e e e e e e de de de e e e e de e e 
```

**Epoca 2**
```
The Project Gutenberg eBook of Dom Casmura de de para de e para a de para de e e e para a de para de e e e para de e e para de e e e parara de e e para de e e para de e e e parara de e e par
```

**Epoca 3**
```
The Project Gutenberg eBook of Dom Casmuitú de para de mas a mas de para ma mas de para ma mas de para ma mas de para ma mas de para ma mas de para ma mas de para ma mas de para ma mas de pa
```

**Epoca 4**
```
The Project Gutenberg eBook of Dom Casmustos de a como de a como a mesta de a mas com a mas de mestas a comos a mestas de a mas com a mas de mestas a com a mas de mestas a com a mas de mesta
```

**Epoca 5**
```
The Project Gutenberg eBook of Dom Casmuito de para de como de para de para de para de para de para de para de para de para de para de para de para de para de para de para de para de para de
```

**Epoca 6**
```
The Project Gutenberg eBook of Dom Casmus de com a minha de minha mãe a mãe a mãe de me a minha de mãe a mãe a me me a minha de mas para de minha mãe a mãe a mãe de me a minha de mãe a mãe a
```

**Epoca 7**
```
The Project Gutenberg eBook of Dom Casmuito de minha mãe de minha mãe de menha mãe de menha mãe esta de menha mais de minha mãe e a meu para de minha mãe e a menha de minha mãe de minha mãe 
```

**Epoca 8**
```
The Project Gutenberg eBook of Dom Casmulo de contario de com a minha de minha de minha e a mãe de contar a minha de contar a minha de contar a minha de contar a minha de contar a minha de c
```

**Epoca 9**
```
The Project Gutenberg eBook of Dom Casmuras de com a minha de minha mãe de minha mãe de minha mãe de minha mãe de minha mãe de minha mãe de minha mãe de minha mãe de minha mãe de minha mãe d
```

**Epoca 10**
```
The Project Gutenberg eBook of Dom Casmura a para de minha mãe
achara de minha mãe de contar a minha mãe de contar a minha mãe de
contar e a minha mãe de contar a minha mãe de contar a minha
```
