# Análise dos Resultados — Sprint 4

Configuração base: `vocab_size=1000, context_length=64, emb_dim=128,
n_heads=4, n_layers=4, drop_rate=0.1, ff_dim=4·emb_dim`. Cada experimento
altera **um** parâmetro por vez.

> **Sobre os números:** as contagens de parâmetros abaixo foram calculadas
> pela fórmula do glossário (conferida contra a soma manual de um modelo
> em NumPy e contra o GPT-2 pequeno: 124.412.160 ≈ 124M). **Os tempos de
> forward dependem da sua máquina:** rode `experiments/run_experiments.py`,
> que salva os valores medidos em `experiments/RESULTADOS_EXPERIMENTOS.md`,
> e use-os para complementar as tabelas.

## Exp. 1 — Número de Transformer Blocks

| n_layers | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| Parâmetros | 462.336 | 660.224 | 1.056.000 | 1.847.552 | 2.639.104 |

Cada bloco adiciona exatamente **197.888** parâmetros (diferença constante).
O crescimento é **linear** em `n_layers`; o tempo de forward também tende a
crescer linearmente, pois os blocos são executados em sequência. Mais blocos
dão mais capacidade de refinar a representação, ao custo de mais memória e
tempo.

## Exp. 2 — Número de heads

Com D=128 fixo, `n_heads` ∈ {1, 2, 4, 8, 16} mantém **1.056.000** parâmetros
em todos os casos: Wq, Wk, Wv e `out_proj` continuam `D×D`. O que muda é
`d_h = D/H` (128, 64, 32, 16, 8). Mais heads = mais "perspectivas" de
atenção em paralelo, porém cada uma com dimensão menor. O custo em
parâmetros é o mesmo; o formato das matrizes de atenção `(B,H,T,T)` cresce
com H (mais memória de ativação).

## Exp. 3 — Dimensão dos embeddings

| emb_dim | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| Parâmetros | 116.544 | 331.392 | 1.056.000 | 3.684.864 |

Dobrar D de 128 para 256 multiplica os parâmetros por ≈ 3,5. O crescimento
se aproxima de **quadrático**, pois os blocos escalam com D² (atenção e FFN),
enquanto embeddings e cabeça de saída escalam linearmente com D.

## Exp. 4 — Dimensão da FeedForward

| ff_dim | 128 | 256 | 512 (=4D) | 1024 |
|---|---|---|---|---|
| Parâmetros | 661.248 | 792.832 | 1.056.000 | 1.582.336 |

A FFN responde por ≈ **67%** dos parâmetros de cada bloco (131.712 contra
65.664 da atenção, com D=128). Por isso `ff_dim` é uma das alavancas mais
fortes de custo do modelo; o crescimento é linear em `ff_dim`.

## Exp. 5 — Tamanho do contexto

| context_length | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| Parâmetros | 1.049.856 | 1.051.904 | 1.056.000 | 1.064.192 |

Só a tabela de embeddings posicionais cresce (`context_length × D`), então
o impacto em parâmetros é pequeno. O custo real do contexto aparece na
**computação**: a matriz de atenção é `T×T`, logo tempo e memória de ativação
crescem aproximadamente de forma **quadrática** com o comprimento da
sequência processada.

## Exp. 6 — Tamanho do vocabulário

| vocab_size | 500 | 1.000 | 5.000 | 20.000 |
|---|---|---|---|---|
| Parâmetros | 928.000 | 1.056.000 | 2.080.000 | 5.920.000 |

V afeta a tabela de embeddings (`V×D`) e a cabeça de saída (`D×V`): cada
token adicional custa `2D` parâmetros. Com V=20.000, essas duas camadas
somam 5,1 M dos 5,9 M de parâmetros: em vocabulários grandes, elas dominam
o modelo (no GPT-2 pequeno, a cabeça sozinha tem ≈ 38,6 M). Daí a
vantagem do *weight tying*.

## Exp. 7 — Parâmetros de geração (temperatura e top-k)

Resultados medidos no arquivo `RESULTADOS_EXPERIMENTOS.md`. O comportamento
esperado, pela teoria e pela validação em NumPy:

- **Greedy** (temperatura 0): determinístico, mesma saída em toda execução.
- **Temperatura baixa**: concentra a probabilidade nos tokens mais prováveis
  (entropia menor); **alta**: achata a distribuição (entropia maior).
- **Top-k pequeno**: só k tokens podem ser sorteados, reduzindo a diversidade;
  `top_k=1` equivale ao greedy.

Como o modelo não foi treinado, a distribuição inicial é quase uniforme,
então o efeito de temperatura e top-k fica mais evidente na **diversidade
de tokens** do que na "qualidade" do texto.

## Exp. 8 — LayerNorm e conexões residuais

A LayerNorm leva qualquer vetor a média ≈ 0 e desvio ≈ 1 (verificado: um
vetor com média 7 e desvio 3 foi normalizado corretamente). Com LayerNorm e
conexões residuais, a escala dos logits permanece estável conforme se
empilham mais blocos (ver tabela medida no arquivo de resultados).

## Exp. 9 — Dropout

Em modo `train`, duas passagens idênticas divergem (ativações zeradas
aleatoriamente); em modo `eval` o resultado é determinístico. Dropout não
altera o número de parâmetros e quase não altera o custo.

## Conclusão geral

1. **Custo em parâmetros:** domina o `emb_dim` (D²), seguido de `ff_dim` e
   `n_layers` (linear); `n_heads` e `drop_rate` não alteram a contagem.
2. **Custo em computação:** cresce com o número de blocos (linear) e com o
   comprimento da sequência (quadrático, por causa da atenção T×T).
3. **Estabilidade:** LayerNorm e conexões residuais permitem empilhar blocos
   sem que a escala das ativações exploda.
4. **Geração:** a qualidade e a diversidade dependem tanto do modelo quanto
   da estratégia de seleção (greedy, temperatura, top-k). Para texto com
   sentido, seria necessário **treinar** o modelo (próxima etapa do livro).
