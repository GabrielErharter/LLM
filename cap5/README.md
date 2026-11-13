# Sprint 5 — Treinamento e Avaliação do GPT

Treina o GPT da Sprint 4 (previsão do próximo token), avalia com
cross-entropy e perplexity, compara configurações de treino e estratégias de
geração, e salva/carrega os pesos.

## Estrutura

```
sprint5/
├── src/
│   ├── gpt.py            # modelo (Sprint 4; geração com top-k ajustada)
│   ├── dados.py          # leitura do texto, split treino/val, DataLoader
│   ├── treino.py         # loss, avaliação, laço de treino, salvar/carregar
│   ├── pipeline.py       # funções de alto nível, geração e gráficos
│   ├── tokenizador.py    # tokenizadores char e palavra (sem PyTorch)
│   └── metricas.py       # perplexity, distinct-n, repetição, sobreajuste (sem PyTorch)
├── experiments/
│   ├── exp_treino.py     # lr, lote, dropout, épocas
│   └── exp_geracao.py    # greedy, temperatura, top-k (treinado x não treinado)
├── data/
│   ├── amostra.txt       # texto minúsculo, só para teste de fumaça
│   └── texto.txt         # (você adiciona) texto de treino de verdade
├── treinar.py            # treino base + avaliação + checkpoint + prova de recarga
├── carregar_e_gerar.py   # inferência a partir do checkpoint
├── baseline_bigrama.py   # régua estatística (sem PyTorch)
├── glossario-capitulo-5.md
├── DOCUMENTACAO_TECNICA.md
└── README.md
```

## Como executar

Primeiro, **coloque um texto em `data/texto.txt`** (um livro em português de
domínio público, em `.txt`, idealmente ≥ 200 mil caracteres). Sem ele, tudo
roda com a amostra, só como teste.

```bash
pip install matplotlib
python baseline_bigrama.py
python treinar.py                       # gera checkpoints/ (modelo, histórico, gráfico)
python carregar_e_gerar.py --prompt "Era uma vez"
python experiments/exp_treino.py --rapido   # teste rápido (2 épocas)
python experiments/exp_treino.py            # completo (demora mais)
python experiments/exp_geracao.py
```

Dicas: `--epocas N` controla o tempo de treino; `--tokenizador palavra` usa
tokens por palavra (precisa de texto maior que o contexto); o `.pt` não deve ir
para o GitHub se for grande (adicione `checkpoints/*.pt` ao `.gitignore`, ou
suba apenas se o professor pedir).

## Entregáveis

| # | Entregável | Onde |
|---|---|---|
| 1 | Glossário | `glossario-capitulo-5.md` |
| 2 | Avaliação (loss, perplexity) | `src/treino.py`, `treinar.py`, `src/metricas.py` |
| 3 | Treinamento | `src/treino.py`, `treinar.py` |
| 4 | Experimentos de treino | `experiments/exp_treino.py` → `RESULTADOS_TREINO.md` |
| 5 | Estratégias de geração | `experiments/exp_geracao.py` → `RESULTADOS_GERACAO.md` |
| 6 | Salvar/carregar pesos | `src/treino.py`, `treinar.py`, `carregar_e_gerar.py` |
| 7 | Documentação técnica | `DOCUMENTACAO_TECNICA.md` |

## Observação sobre validação

Aqui não havia PyTorch no ambiente de desenvolvimento, então o código de
PyTorch **não foi executado antes da entrega**. Foram validados em Python
puro/NumPy: a fórmula da cross-entropy e seu gradiente (conferido por derivada
numérica), perplexity = exp(loss), loss inicial = ln(V), tokenizadores,
divisão sem vazamento, métricas e detecção de sobreajuste. Se algum erro
aparecer ao rodar, envie a mensagem para correção.
