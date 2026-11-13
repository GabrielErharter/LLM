# Documentação Técnica — Sprint 5: Treinamento e Avaliação do GPT

> **Como usar este documento:** as seções 1 a 9 descrevem o procedimento e já
> estão completas. A seção 10 deve ser preenchida com os resultados gerados na
> sua máquina (os scripts criam os arquivos `RESULTADOS_*.md` automaticamente),
> e a seção 11 ajuda a interpretá-los.

## 1. Visão geral e como reproduzir

Treina-se o GPT da Sprint 4 para prever o próximo token em um texto, mede-se a
qualidade com cross-entropy e perplexity, comparam-se configurações de treino e
estratégias de geração, e salva-se o modelo.

Ordem sugerida de execução (dentro da pasta `sprint5/`):

```bash
pip install matplotlib                                  # uma vez
python baseline_bigrama.py                              # régua (sem PyTorch)
python treinar.py                                       # treino base + checkpoint
python carregar_e_gerar.py --prompt "Era uma vez"       # inferência com os pesos salvos
python experiments/exp_treino.py --rapido               # teste de fumaça
python experiments/exp_treino.py                        # experimentos completos
python experiments/exp_geracao.py                       # estratégias de decodificação
```

Reprodutibilidade: sementes fixas (`--seed 123`), parâmetros gravados no
checkpoint (`extra.params`) e no `historico_base.json`. Registre também a versão
do PyTorch (`python -c "import torch; print(torch.__version__)"`) e o dispositivo
(CPU/GPU, impresso no início do treino).

## 2. Dados

- Arquivo de treino: `data/texto.txt` (use um livro de domínio público em
  português, idealmente com 200 mil caracteres ou mais). Sem ele, o código usa
  `data/amostra.txt` (≈ 2,4 mil caracteres), **apenas como teste de fumaça**.
- **Tokenização:** `char` (padrão: 1 token = 1 caractere; vocabulário pequeno e
  treino rápido em CPU) ou `palavra` (igual à Sprint 1, com `<|unk|>`). O
  vocabulário é construído sobre o texto inteiro.
- **Divisão treino/validação:** 90% / 10% por posição, sem embaralhar antes.
- **Janelas:** contexto de 64 tokens, `stride` = contexto (sem sobreposição);
  entrada = tokens `[i, i+T)`, alvo = `[i+1, i+T+1)`.

## 3. Configuração base

| Parâmetro | Valor | | Parâmetro | Valor |
|---|---|---|---|---|
| Épocas | 10 | | Camadas (blocos) | 4 |
| Tamanho do lote | 16 | | Heads | 4 |
| Contexto | 64 | | Dimensão do embedding | 128 |
| Otimizador | AdamW | | FeedForward | 4 × 128 = 512 |
| Taxa de aprendizado | 5e-4 | | Dropout | 0,1 |
| Weight decay | 0,1 | | Gradient clipping | desligado |
| Semente | 123 | | Avaliação durante o treino | a cada 20 passos, 5 lotes |

## 4. Procedimento de treinamento (`src/treino.py`)

Para cada época e cada lote:

1. `optimizer.zero_grad()`: zera os gradientes acumulados.
2. Forward: `logits = model(entrada)` → forma `(B, T, V)`.
3. Loss: `F.cross_entropy(logits.flatten(0,1), alvo.flatten())`.
4. `loss.backward()`: retropropagação, calculando o gradiente de cada peso.
5. (Opcional) recorte do gradiente.
6. `optimizer.step()`: o AdamW atualiza os pesos.

Ao fim de cada época avalia-se o conjunto inteiro (treino e validação, em modo
`eval`, sem dropout e sem gradientes) e gera-se uma amostra greedy com o mesmo
prompt. Antes de treinar registra-se o **estado inicial**.

## 5. Avaliação

- **Cross-entropy:** média de `−log p(token correto)` sobre todas as posições.
- **Perplexity:** `exp(loss)`; lê-se como o número efetivo de tokens entre os
  quais o modelo hesita.
- **Referências:** modelo aleatório → `loss ≈ ln(V)` e `perplexity ≈ V`
  (verificado numericamente); modelo de bigramas (`baseline_bigrama.py`) como
  régua estatística; um modelo perfeito teria perplexity 1.
- Comparar sempre **treino x validação** e **inicial x final**.

## 6. Experimentos de treinamento (`experiments/exp_treino.py`)

Um parâmetro por vez, a partir da base: **taxa de aprendizado** (1e-4, 5e-4,
2e-3, 1e-2), **tamanho do lote** (4, 16, 64), **dropout** (0, 0,1, 0,3) e **número
de épocas** (lido da curva do treino base: sem scheduler e com semente fixa, o
treino de E épocas é o prefixo do treino de E_máx épocas). Registra-se:
configuração, loss de treino e validação, perplexity, diferença val − treino,
melhor época, sinal de sobreajuste, tempo, tokens/s e amostra de texto.
Saídas: `RESULTADOS_TREINO.md`, `resultados_treino.json`, `figuras/treino_*.png`.

## 7. Estratégias de geração (`experiments/exp_geracao.py`)

Mesmo prompt em todas as gerações. Estratégias: greedy; amostragem com
temperatura 0,5 / 1,0 / 1,5; top-k 5 e 20 (T = 1,0); top-k 5 com T = 0,5.
Amostradas com 5 sementes; greedy é determinístico. Métricas sobre palavras:
distinct-1 e distinct-2 (diversidade), repetição de trigramas (loops) e fração
de palavras que existem no texto de treino (proxy de coerência ortográfica).

**Separando estratégia e treino:** repete-se a comparação em um modelo **não
treinado** (mesma arquitetura, pesos aleatórios). Diferenças entre linhas do
mesmo modelo vêm da estratégia; diferenças entre o modelo treinado e o não
treinado na mesma estratégia vêm dos pesos.

## 8. Salvamento e carregamento

`salvar_checkpoint` grava em um único arquivo (`checkpoints/modelo_base.pt`):

| Chave | Conteúdo | Para quê |
|---|---|---|
| `model_state_dict` | pesos de todas as camadas | inferência |
| `optimizer_state_dict` | estado do AdamW | retomar o treino |
| `config` | vocab, contexto, emb, heads, camadas, dropout, ff | reconstruir a arquitetura |
| `tokenizador` | tipo e vocabulário | codificar/decodificar igual ao treino |
| `historico` | losses por passo e por época, amostras | registros |
| `extra` | parâmetros do treino, prompt, arquivo de dados | reprodução |

`carregar_checkpoint` reconstrói `GPTModel(config)`, carrega os pesos e coloca em
`eval()`. O `treinar.py` **prova** o procedimento ao final: o modelo carregado
produz os mesmos logits que o modelo em memória (`allclose`) e gera texto.
Para retomar: `python treinar.py --retomar checkpoints/modelo_base.pt --epocas 5`.
Para só gerar: `python carregar_e_gerar.py --temperatura 0.8 --top-k 20`.

## 9. Limitações

- Modelo e texto pequenos e **treinamento curto, em CPU**; o texto gerado
  melhora em ortografia e estrutura, mas não deve ser esperado como coerente.
- Tokenização por caractere (ou por palavra, com `<unk>`), diferente do BPE do
  GPT-2; os valores de loss/perplexity **não são comparáveis** entre
  tokenizadores nem com os do livro.
- Vocabulário construído sobre o texto inteiro (inclusive validação): efeito
  desprezível por caractere, mas existe.
- Divisão única treino/validação, uma semente por configuração: diferenças
  pequenas entre execuções podem ser ruído.
- Sem scheduler de taxa de aprendizado, sem warmup e sem recorte de gradiente
  na base; hiperparâmetros não foram otimizados.
- As métricas de geração (distinct-n, palavras válidas) são proxies simples e
  não medem coerência semântica.

## 10. Resultados (preencher com os arquivos gerados)

- **Treino base:** `checkpoints/RESULTADOS_TREINO_BASE.md` e `checkpoints/curva_loss_base.png`.
- **Experimentos de treino:** `experiments/RESULTADOS_TREINO.md` e `experiments/figuras/`.
- **Estratégias de geração:** `experiments/RESULTADOS_GERACAO.md`.
- **Baseline:** saída de `python baseline_bigrama.py`.

Resumo a registrar aqui (copie os números reais):

| Item | Valor |
|---|---|
| Texto, nº de tokens, vocabulário | |
| Parâmetros do modelo / dispositivo / tempo de treino | |
| Loss e perplexity iniciais (treino / val) | |
| Loss e perplexity finais (treino / val) | |
| Baseline: ln(V) e bigramas (val) | |
| Melhor época pela validação; sobreajuste? | |

## 11. Guia de interpretação

| O que você observa | Leitura provável |
|---|---|
| Loss inicial próxima de ln(V) | Esperado: o modelo começa "chutando" |
| Treino e validação caem juntas | O modelo está aprendendo e generalizando |
| Treino cai, validação sobe | Sobreajuste (texto pequeno, muitas épocas, pouca regularização) |
| Loss quase não cai com lr baixo | Convergência lenta: aumentar lr ou épocas |
| Loss oscila ou explode com lr alto | Passos grandes demais: instabilidade/divergência |
| Lote maior → menos passos por época | Mesma época, menos atualizações: pode precisar de lr maior |
| Dropout maior → gap val−treino menor, loss de treino maior | Regularização funcionando (às custas de ajuste no treino) |
| Greedy repete trechos | Típico: sem aleatoriedade o modelo cai em loops |
| T alta ou top-k grande → mais "palavras inválidas" | Mais diversidade com perda de coerência |
| Modelo não treinado gera lixo em qualquer estratégia | A estratégia não cria conhecimento: ele vem dos pesos |
| Loss de validação do GPT acima da do bigrama | Ainda não superou a estatística trivial (dados pequenos? poucas épocas?) |
