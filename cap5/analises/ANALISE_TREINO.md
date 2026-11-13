# Análise — Avaliação e Experimentos de Treinamento (Entregáveis 2, 3 e 4)

Fonte dos números: `experiments/RESULTADOS_TREINO.md` (execução real).

## 1. Dados e configuração

- Texto: `texto.txt`, 400.944 caracteres (livro *Dom Casmurro*, versão do Projeto
  Gutenberg). Tokenização por caractere: **V = 112**.
- Divisão 90% / 10% por posição: ≈ 360.849 tokens de treino (≈ 5.638 janelas de
  64 tokens) e ≈ 40 mil de validação.
- Modelo: 4 blocos, 4 heads, emb 128, FeedForward 512, dropout 0,1 →
  **828.672 parâmetros** (calculado pela fórmula do glossário).
- Treino base: AdamW, lr = 5e-4, weight decay 0,1, lote 16, 10 épocas
  (≈ 3.520 passos; ≈ 3,6 milhões de tokens vistos), semente 123.

## 2. Avaliação: estado inicial e final

| | loss treino | loss val | ppl treino | ppl val |
|---|---|---|---|---|
| Chute uniforme: ln(112) | 4,7185 | 4,7185 | 112 | 112 |
| Inicial (antes de treinar) | 4,8692 | 4,7966 | 130,2 | 121,1 |
| Final (época 10) | 1,5986 | 2,7069 | 4,95 | 15,0 |
| Melhor validação (época 9) | 1,6315 | 2,6920 | 5,1 | 14,8 |

- O modelo começa próximo do chute uniforme (loss ≈ ln V), como esperado de
  pesos aleatórios (um pouco acima, pois a inicialização não é exatamente uniforme).
- Ao final, a perplexity de validação cai de ≈ 121 para ≈ 15: o modelo deixa de
  hesitar entre ~120 caracteres e passa a hesitar, em média, entre ~15. Isto é a
  evidência quantitativa de que ele **está aprendendo** a prever o próximo token.

## 3. Experimento D — Número de épocas (treino base)

| época | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| loss treino | 2,355 | 2,147 | 2,007 | 1,910 | 1,836 | 1,766 | 1,714 | 1,668 | 1,632 | 1,599 |
| loss val | 3,390 | 3,216 | 3,061 | 2,910 | 2,851 | 2,784 | 2,713 | 2,745 | 2,692 | 2,707 |
| ppl val | 29,7 | 24,9 | 21,4 | 18,3 | 17,3 | 16,2 | 15,1 | 15,6 | 14,8 | 15,0 |

- A maior parte do ganho ocorre nas 4 primeiras épocas (a loss de validação cai
  ≈ 0,15 a 0,17 por época); depois os ganhos diminuem (≈ 0,06 a 0,07 por época)
  e a validação passa a oscilar (sobe em 8 e em 10).
- A loss de **treino continua caindo** até a época 10 (≈ −0,03 por época): o
  modelo ainda não convergiu; mais épocas reduziriam mais a loss de treino.
- **Evolução do texto (greedy, mesmo prompt):** época 1 gera apenas
  `de de de e e e` (os tokens mais frequentes); época 5, palavras reais em loop
  (`de para de para`); época 10, trechos com vocabulário do livro
  (`minha mãe de contar a minha mãe`), ainda em loop.

## 4. Experimento A — Taxa de aprendizado

| lr | loss treino | loss val | ppl val | tempo (s) |
|---|---|---|---|---|
| 1e-4 | 2,090 | 3,217 | 25,0 | 297,9 |
| 5e-4 (base) | 1,599 | 2,707 | 15,0 | 268,8 |
| **2e-3** | **1,434** | **2,563** | **13,0** | 263,6 |
| 1e-2 | 1,688 | 2,650 | 14,2 | 274,6 |

- **lr = 1e-4 é lento demais:** em 10 épocas a loss fica bem acima (val 3,217) e
  o texto ainda é `de a de por de a`.
- **lr = 2e-3 foi o melhor** dos quatro (ppl val 13,0 contra 15,0 do base).
- **lr = 1e-2 não divergiu**, mas ficou pior que 2e-3 tanto no treino (1,688
  contra 1,434) quanto na validação: passos grandes demais deixam a otimização
  mais ruidosa. Um modelo pequeno e com AdamW tolera lr alto, mas há um ótimo
  intermediário.
- O custo computacional é praticamente o mesmo (264 a 298 s): a taxa de
  aprendizado muda a qualidade, não o custo.

## 5. Experimento B — Tamanho do lote (lr fixo em 5e-4)

| lote | passos/época | loss treino | loss val | ppl val | tempo (s) | tokens/s |
|---|---|---|---|---|---|---|
| 4 | ≈ 1.409 | 1,505 | 2,580 | 13,2 | 483,0 | 7.468 |
| 16 | ≈ 352 | 1,599 | 2,707 | 15,0 | 271,6 | 13.270 |
| 64 | ≈ 88 | 1,844 | 2,856 | 17,4 | 286,9 | 12.563 |

- Com o mesmo número de **épocas**, lote menor significa **mais atualizações de
  pesos** (≈ 14 mil contra ≈ 3,5 mil e ≈ 880 passos), e por isso a loss final foi
  menor. O resultado mistura o efeito do lote com o do número de passos, já que
  o lr foi mantido igual.
- **Custo:** o lote 4 levou 1,8 vezes mais tempo que o 16 e processou menos
  tokens por segundo (7,5 mil contra 13,3 mil), pois cada passo tem um custo fixo
  e usa pouco paralelismo. O lote 64 **não foi mais rápido** que o 16 neste
  computador (287 s contra 272 s): sem ganho de velocidade com lotes maiores aqui.
- Texto: o lote 4 já mostra construções mais próximas do livro
  (`Capitú`, `minha mãe era mãe`); o lote 64 segue em loops de palavras curtas.

## 6. Experimento C — Dropout

| dropout | loss treino | loss val | ppl val | val − treino | tempo (s) |
|---|---|---|---|---|---|
| 0 | 1,438 | **2,667** | **14,4** | +1,229 | **206,1** |
| 0,1 (base) | 1,599 | 2,707 | 15,0 | +1,108 | 275,5 |
| 0,3 | 1,839 | 2,828 | 16,9 | +0,989 | 263,5 |

- O dropout cumpre o papel de regularizar: **a diferença val − treino diminui**
  (1,229 → 1,108 → 0,989), mas ao custo de uma loss de treino maior (1,438 →
  1,839).
- Dentro de 10 épocas **a validação foi melhor sem dropout** (14,4 contra 15,0
  e 16,9). O modelo ainda está subajustado, não superajustado; regularizar
  mais só desacelera o aprendizado.
- Também afetou o tempo: com dropout = 0 o treino levou ≈ 25% menos tempo
  (206 s contra 275 s).

## 7. Sobreajuste e diferença entre treino e validação

**O que o rastreador marcou:** `sobreajuste = SIM` para lr = 2e-3 e para
dropout = 0,3, com melhor época 9 e piora na época 10. Essa evidência é **fraca**:

- Foi uma alta de uma única época, de poucos centésimos, comparável à oscilação
  que o próprio treino base também mostra (a validação sobe nas épocas 8 e 10).
- Marcou SIM justamente em dropout = 0,3, a configuração com **mais**
  regularização, onde o sobreajuste seria menos esperado, o que indica que o
  critério é sensível a ruído.
- Nos demais casos a melhor época é 9 ou 10 (a validação ainda estava
  melhorando no fim).

**A diferença val − treino é grande e quase constante:** ≈ +1,0 desde a
**época 1** (1,035) até a época 10 (1,108). No sobreajuste clássico, a diferença
**cresce** ao longo das épocas; aqui ela já aparece inteira na primeira época e
quase não muda. Isso é mais compatível com os dois conjuntos terem **distribuições
diferentes** do que com memorização. Hipótese a verificar: o arquivo do Projeto
Gutenberg inclui, no início, um cabeçalho e, no fim, uma licença longa em
inglês; como a validação é o **último 10%** do texto, ela pode conter grande
parte dessa licença, que o modelo (treinado em português) quase nunca viu.

**Como verificar:** abra o fim de `data/texto.txt`. Se aparecer
`*** END OF THE PROJECT GUTENBERG EBOOK ...` e páginas de texto em inglês,
apague tudo antes de `*** START OF ...` e depois de `*** END OF ...` e refaça o
treino. Compare também com `python baseline_bigrama.py`: se o baseline de
bigramas também tiver loss de validação muito acima da de treino, a causa é a
diferença entre os conjuntos, e não o modelo.

## 8. Custo computacional

| Configuração | tempo (s) | tokens/s | observação |
|---|---|---|---|
| base (lr 5e-4, lote 16, dropout 0,1) | 268,8 / 271,6 / 275,5 | ≈ 13.100 a 13.400 | repetida em 3 experimentos |
| lote 4 | 483,0 | 7.468 | 1,8× o tempo |
| lote 64 | 286,9 | 12.563 | sem ganho de velocidade |
| dropout 0 | 206,1 | 17.489 | ≈ 25% mais rápido |
| lr (4 valores) | 264 a 298 | 12.100 a 13.700 | lr não altera o custo |

A configuração base foi executada três vezes (nos experimentos A, B e C) e deu
**exatamente** a mesma loss (1,5986 / 2,7069): com semente fixa, o treino é
reprodutível. O tempo variou apenas ≈ 3%, o que dá uma ideia do ruído de medição.

## 9. Conclusões e limitações

1. O modelo **aprende**: a perplexity de validação cai de ≈ 121 (próxima de V)
   para ≈ 15, e o texto evolui de tokens frequentes para palavras e frases do
   livro.
2. Dentre as configurações testadas, **lr = 2e-3** foi a melhor (ppl val 13,0), e o
   **lote 4** também melhorou a loss, ao custo de 1,8× o tempo.
3. O treino é **subajustado**: a loss de treino ainda caía na época 10, e
   regularizar mais (dropout 0,3) piorou a validação. Mais épocas e/ou lr maior
   são o caminho natural.
4. A diferença treino-validação (≈ +1,0) **não deve ser lida como sobreajuste**
   sem antes verificar a hipótese do cabeçalho/licença em inglês (seção 7).
5. Limitações: modelo pequeno, uma semente por configuração (diferenças de
   ≈ 0,05 podem ser ruído), um único livro, sem scheduler de lr, tokenização por
   caractere (loss e perplexity **não são comparáveis** com as do livro, que usa
   BPE) e o texto gerado em greedy continua repetitivo e sem coerência global.
