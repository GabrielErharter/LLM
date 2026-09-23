# Análise dos Resultados — Sprint 3 (Attention)

## Experimento 1 — Diferentes dimensões de embedding

À medida que `d_model` aumenta (8 → 16 → 32 → 64), a forma da saída de
contexto acompanha essa dimensão, mas a forma da **matriz de atenção**
permanece `(context_size, context_size)` — a dimensão de embedding não
afeta o tamanho da matriz de atenção, apenas a dimensão dos vetores
Query/Key/Value usados para calculá-la.

## Experimento 2 — Diferentes números de heads

Com `d_model=32` fixo, variar o número de heads (1, 2, 4, 8) muda a
`head_dim` (32, 16, 8, 4 respectivamente), mas **não muda a quantidade
total de parâmetros** do módulo — as matrizes Wq, Wk, Wv continuam
tendo dimensão `d_model × d_model`; o que muda é como essa dimensão é
"fatiada" entre as heads durante o cálculo de atenção.

## Experimento 3 — Diferentes dimensões de head

Ao aumentar `d_model` mantendo `num_heads=4` fixo, a `head_dim` cresce
proporcionalmente (4, 8, 16, 32). Isso mostra que a dimensão de cada
head é sempre `d_model / num_heads` — os dois parâmetros estão
diretamente acoplados.

## Experimento 4 — Self-Attention vs Multi-Head Attention

A Self-Attention simples (sem máscara, sem múltiplas heads) e o
Multi-Head Attention com o mesmo `d_model` têm quantidades de
parâmetros diferentes, pois o Multi-Head Attention possui uma camada
adicional de projeção de saída (`out_proj`). Conceitualmente, a
diferença mais importante não é o número de parâmetros, e sim a
**capacidade representacional**: o Multi-Head permite capturar
múltiplos "tipos" de relação entre tokens simultaneamente (cada head
aprende seus próprios Wq/Wk/Wv), enquanto a Self-Attention simples
captura apenas um único padrão de relação.

## Experimento 5 — Visualização da matriz de atenção

A matriz de atenção causal, visualizada como heatmap, apresenta o
padrão triangular inferior esperado: a diagonal principal e tudo
abaixo dela têm valores não-nulos, enquanto a parte acima da diagonal
é sistematicamente zero. Isso confirma visualmente que a máscara
causal está impedindo o acesso a tokens futuros.

## Experimento 6 — Attention com e sem escala

A escala (`1/√d_k`) tem impacto direto na "suavidade" da distribuição
de atenção. Medindo a entropia média dos pesos de atenção:

| Configuração | Entropia média |
|---|---|
| Sem escala | menor (distribuição mais "dura", próxima de one-hot) |
| Com escala | maior (distribuição mais suave, mais uniforme) |

Isso confirma a justificativa teórica do capítulo: em dimensões altas
(`d_model=64` no experimento), os produtos escalares tendem a ter
variância maior, o que sem escala levaria a poucos tokens dominando
quase toda a atenção — prejudicando o fluxo de gradiente durante o
treinamento.

## Experimento 7 — Comportamento da máscara causal

A soma dos pesos de atenção na região triangular superior (posições
futuras) é numericamente zero, confirmando que a máscara está sendo
aplicada corretamente antes do Softmax (usando `-infinito`, que se
torna 0 após a normalização).

## Experimento 8 — Diferentes sequências de entrada

A mesma camada de atenção, com os mesmos pesos aprendidos, processa
corretamente sequências de tamanhos diferentes (3, 6, 12), desde que
não ultrapassem o `context_length` máximo configurado na inicialização
da camada. Isso é possível porque a máscara causal é pré-computada
para o tamanho máximo e "recortada" (`[:context_size, :context_size]`)
conforme o tamanho real da sequência processada.

## Conclusão geral

Os experimentos confirmam experimentalmente os principais conceitos
teóricos do capítulo:

1. A **escala** (`√d_k`) é necessária para evitar saturação do Softmax
   em dimensões de embedding altas;
2. A **máscara causal** bloqueia efetivamente o acesso a tokens
   futuros, essencial para o treinamento autorregressivo;
3. O **Multi-Head Attention** distribui a mesma capacidade
   representacional em múltiplas "perspectivas" paralelas, sem
   aumentar substancialmente o custo de parâmetros das projeções
   principais;
4. A implementação é **flexível quanto ao tamanho da sequência**,
   respeitando apenas o limite máximo de contexto configurado.
