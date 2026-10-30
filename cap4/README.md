# Sprint 4 — Arquitetura GPT e Geração de Texto

Integra o Multi-Head Attention da Sprint 3 em um modelo GPT simplificado:
Transformer Block (LayerNorm, FeedForward, GELU, conexões residuais),
empilhamento de blocos, Language Model Head e geração autorregressiva.

## Estrutura

```
sprint4/
├── src/
│   └── gpt.py                      # LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel, generate
├── experiments/
│   ├── run_experiments.py          # 9 experimentos (gera RESULTADOS_EXPERIMENTOS.md ao rodar)
│   └── RESULTADOS_EXPERIMENTOS.md  # criado automaticamente na sua máquina
├── DOCUMENTACAO.md                 # Estrutura do bloco, dimensões dos tensores, procedimento de geração
├── ANALISE_RESULTADOS.md           # Análise dos experimentos
├── glossario-capitulo-4.md         # Glossário do Capítulo 4
└── README.md
```

## Como executar (ambiente com PyTorch)

```bash
cd sprint4
python src/gpt.py                       # demonstração: logits, parâmetros e geração
python experiments/run_experiments.py   # experimentos + relatório automático
```

Não precisa instalar nada além do que já está no ambiente (`torch`).

## Entregáveis

| # | Entregável | Onde está |
|---|---|---|
| 1 | Glossário | `glossario-capitulo-4.md` |
| 2 | Transformer Block + documentação | `src/gpt.py` (`TransformerBlock`), `DOCUMENTACAO.md` |
| 3 | Modelo GPT | `src/gpt.py` (`GPTModel`) |
| 4 | Inferência e geração | `src/gpt.py` (`generate`), `DOCUMENTACAO.md` |
| 5 | Experimentação e análise | `experiments/run_experiments.py`, `ANALISE_RESULTADOS.md` |

## Observações

- O modelo **não é treinado** nesta sprint: os pesos são aleatórios, então o
  texto gerado não tem sentido. O objetivo é demonstrar o fluxo
  logits → probabilidades → seleção → geração autorregressiva.
- A lógica (LayerNorm, GELU, formas dos tensores, causalidade, contagem de
  parâmetros e geração) foi validada previamente com NumPy.
