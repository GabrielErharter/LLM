# Análise — Estratégias de Geração (Entregável 5)

Fonte: `experiments/RESULTADOS_GERACAO.md`. Checkpoint `modelo_base.pt`, tokenização
por caractere (V = 112), prompt idêntico em todas as gerações
(`The Project Gutenberg eBook of Dom Casmu`), 300 tokens novos, 5 sementes por
estratégia amostrada (o greedy é determinístico: 1 amostra). O texto gerado
pelo greedy coincide com o da época 10 do experimento D, o que indica que o
checkpoint é o treino base de 10 épocas.

## 1. Resultados

Modelo **treinado**:

| estratégia | distinct-1 | distinct-2 | repetição-3 | palavras válidas |
|---|---|---|---|---|
| greedy | 0,129 | 0,203 | 0,750 | 0,971 |
| temperatura 0,5 | 0,631 | 0,979 | 0,003 | 0,826 |
| temperatura 1,0 | 0,864 | 0,996 | 0,000 | 0,564 |
| temperatura 1,5 | 0,912 | 1,000 | 0,000 | 0,447 |
| top-k 5 (T = 1,0) | 0,742 | 0,987 | 0,000 | 0,718 |
| top-k 20 (T = 1,0) | 0,834 | 0,996 | 0,000 | 0,624 |
| top-k 5 (T = 0,5) | 0,497 | 0,899 | 0,012 | 0,847 |

Modelo **não treinado** (mesma arquitetura, pesos aleatórios):

| estratégia | distinct-1 | distinct-2 | repetição-3 | palavras válidas |
|---|---|---|---|---|
| greedy | 0,100 | 0,111 | 0,888 | 0,070 |
| temperatura 0,5 | 0,915 | 1,000 | 0,000 | 0,219 |
| temperatura 1,0 | 0,937 | 1,000 | 0,000 | 0,191 |
| temperatura 1,5 | 0,959 | 1,000 | 0,000 | 0,189 |
| top-k 5 (T = 1,0) | 0,767 | 0,987 | 0,000 | 0,283 |
| top-k 20 (T = 1,0) | 0,907 | 0,997 | 0,000 | 0,263 |
| top-k 5 (T = 0,5) | 0,731 | 0,986 | 0,000 | 0,232 |

Como ler: distinct-n alto = mais diversidade; repetição-3 alta = loops; "palavras
válidas" = fração das palavras geradas que existem no texto de treino.

## 2. Efeito da estratégia (linhas diferentes, mesmo modelo)

- **Greedy:** gera loops. No modelo treinado, o trecho `a minha mãe de contar`
  se repete até o fim (repetição-3 = 0,750; distinct-1 = 0,129). Em cada passo
  escolhe-se o token mais provável, ele volta ao contexto e reforça o mesmo
  padrão. Atenção: as "palavras válidas" são 0,971 só porque as palavras
  repetidas são válidas; essa métrica deve ser lida junto com a repetição.
- **Temperatura:** é um dial entre previsibilidade e diversidade. De 0,5 para 1,5
  a diversidade sobe (distinct-1: 0,631 → 0,864 → 0,912) e as palavras válidas
  caem (0,826 → 0,564 → 0,447). A repetição já some em T = 0,5 (0,003). Nas
  amostras, T = 0,5 produz frases que parecem português antigo (`para e a
  descustar de Capitú`), e T = 1,5 produz ruído (`Escobaçõe; smas por» Wá`).
- **Top-k:** com T = 1,0, k = 5 melhora a legibilidade (palavras válidas 0,718
  contra 0,564 sem top-k) à custa de diversidade (distinct-1 0,742 contra
  0,864); k = 20 fica no meio (0,624 e 0,834), pois corta apenas a cauda mais
  rara do vocabulário.
- **Melhor compromisso observado:** top-k 5 com T = 0,5 teve as palavras válidas
  mais altas entre as estratégias amostradas (0,847) com repetição baixa
  (0,012). O texto é legível, mas já recai em expressões repetidas
  (`a minha mãe`).

## 3. Efeito dos pesos treinados (mesma estratégia, treinado x não treinado)

- **Palavras válidas** (T = 0,5): 0,826 contra 0,219; (top-k 5, T = 0,5): 0,847
  contra 0,232. O modelo não treinado produz símbolos aleatórios
  (`L,zò_i;èIjFQD(=xm`), enquanto o treinado produz sequências de caracteres
  e palavras do livro. O valor ≈ 0,2 do não treinado está inflado por fragmentos
  de um caractere que por acaso são "palavras" do texto.
- **Diversidade:** na mesma temperatura o modelo treinado é *menos* diverso
  (distinct-1 0,631 contra 0,915 em T = 0,5): o treino concentrou a
  probabilidade em caracteres plausíveis, enquanto os pesos aleatórios geram
  quase uniformemente. Mais diversidade, portanto, não significa texto melhor.
- **Sensibilidade à estratégia:** no modelo treinado, as palavras válidas variam
  de 0,826 a 0,447 conforme a temperatura; no não treinado ficam todas em ≈ 0,19
  a 0,28, sem tendência clara. A estratégia só modula o que os pesos
  aprenderam; sem conhecimento nos pesos, não há o que modular.

## 4. Efeitos que pertencem à estratégia, não ao treino

O **loop do greedy aparece nos dois modelos**: `a minha mãe de contar` no
treinado (repetição 0,750) e `_”a_”a_”a` no não treinado (repetição 0,888). A
degeneração vem do `argmax` determinístico realimentando o contexto; do treino
vem apenas **qual** padrão se repete e se ele parece língua humana. O top-k 5 em
T = 1,0 no não treinado também gera motivos repetitivos (`”r_”a`), porque
restringe a escolha a poucos caracteres.

## 5. Diversidade, repetição e coerência (síntese)

| Estratégia | diversidade | repetição | legibilidade | comentário |
|---|---|---|---|---|
| greedy | muito baixa | muito alta | palavras reais, em loop | determinístico |
| T = 0,5 | média | quase nula | boa | bom equilíbrio |
| T = 1,0 / 1,5 | alta / muito alta | nula | cai muito | vira ruído em T = 1,5 |
| top-k 5, T = 1,0 | média-alta | nula | razoável | corta a cauda |
| top-k 5, T = 0,5 | média-baixa | baixa | a melhor | começa a repetir expressões |

**Coerência:** em nenhuma estratégia o texto é globalmente coerente (não há
sentido de frase ou de parágrafo), apenas ortografia e vocabulário plausíveis do
livro, inclusive a grafia antiga (`Tambem`, `elle`). Isso é esperado de um modelo
de ≈ 0,83 milhão de parâmetros treinado em ≈ 360 mil caracteres.

## 6. Limitações

- O prompt é o cabeçalho em inglês do Projeto Gutenberg, pouco natural para um
  modelo treinado em português; vale repetir com um prompt do próprio livro
  (por exemplo, `Capitu`).
- As métricas são calculadas sobre palavras (proxy para um modelo por caractere),
  não medem coerência semântica e são enganosas em saídas degeneradas.
- 5 sementes e um único prompt: diferenças pequenas podem ser ruído.
- A comparação é entre configurações de um único treino de 10 épocas, ainda
  subajustado (ver `ANALISE_TREINO.md`).
