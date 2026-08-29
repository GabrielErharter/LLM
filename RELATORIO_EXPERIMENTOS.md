# Relatório de Experimentos

Resultados obtidos ao executar `experiments/run_experiments.py`.

## Experimento 1 — Quantidade de tokens por tamanho de texto

| Texto | Caracteres | Tokens gerados |
|---|---|---|
| curto | 46 | 12 |
| médio | 191 | 39 |
| longo | 421 | 80 |

**Observação:** a quantidade de tokens cresce de forma aproximadamente
proporcional ao número de caracteres, já que a tokenização usada é
baseada em palavras/pontuação.

## Experimento 2 — Relação entre tamanho do contexto e quantidade de amostras

Texto "longo" (80 tokens totais), stride = context_size:

| context_size | Amostras geradas |
|---|---|
| 2 | 39 |
| 4 | 19 |
| 8 | 9 |
| 16 | 4 |

**Observação:** quanto maior o contexto, menos amostras são geradas —
relação aproximadamente inversa. Cada amostra "consome" mais tokens da
sequência original, restando menos posições de início disponíveis.

## Experimento 3 — Impacto do stride na quantidade de amostras

Texto "longo", context_size = 4 fixo:

| stride | Amostras geradas |
|---|---|
| 1 | 76 |
| 2 | 38 |
| 4 | 19 |

**Observação:** stride menor gera muito mais amostras, pois as janelas
se sobrepõem mais — porém aumenta a redundância entre amostras
consecutivas (compartilham a maior parte dos tokens).

## Experimento 4 — Dimensões das estruturas produzidas pelo pipeline

| batch_size | context_size | embedding_dim | Shape final (batch, ctx, dim) | Nº de valores |
|---|---|---|---|---|
| 2 | 4 | 8 | (2, 4, 8) | 64 |
| 4 | 4 | 16 | (4, 4, 16) | 256 |
| 4 | 8 | 32 | (4, 8, 32) | 1.024 |
| 8 | 16 | 64 | (8, 16, 64) | 8.192 |

**Observação:** o custo computacional (quantidade de valores na entrada
do modelo) cresce proporcionalmente ao produto
`batch_size × context_size × embedding_dim`. Dobrar qualquer uma das
três dimensões dobra o custo total.

## Experimento 5 — Custo de processamento (tempo) por tamanho de texto

Cada texto foi repetido 200x para ampliar a escala e tornar a diferença de tempo mensurável:

| Texto (x200) | Tokens gerados | Tempo de tokenização |
|---|---|---|
| curto | 2.400 | 0,50 ms |
| médio | 7.800 | 1,70 ms |
| longo | 16.000 | 4,16 ms |

**Observação:** o tempo de tokenização cresce de forma aproximadamente
linear com a quantidade de tokens processados — coerente com o fato de
a tokenização percorrer o texto uma única vez.

## Conclusões gerais

- Existe um **trade-off direto entre tamanho de contexto e quantidade de
  amostras de treinamento**: contextos maiores dão ao modelo mais
  informação por amostra, mas geram menos amostras a partir do mesmo
  texto.
- O **stride** é o principal controle de quanta sobreposição/redundância
  existe entre amostras consecutivas.
- O **custo computacional** da entrada do modelo (tensor final) escala
  multiplicativamente com batch_size, context_size e embedding_dim —
  ou seja, aumentar qualquer uma dessas dimensões impacta diretamente a
  quantidade de memória e processamento necessários.
- A tokenização em si é uma etapa computacionalmente barata (poucos
  milissegundos mesmo para milhares de tokens), o que a torna um bom
  candidato para pré-processamento em lote antes do treinamento.
