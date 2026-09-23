# Sprint 3 — Mecanismo de Attention

Implementação incremental do mecanismo central da arquitetura
Transformer: Self-Attention → Scaled Dot-Product Attention → Causal
Attention → Multi-Head Attention.

## Estrutura dos arquivos

```
.
├── src/
│   └── attention.py           # Todas as implementações (itens 2 a 5)
├── experiments/
│   ├── run_experiments.py     # Experimentos + geração das visualizações
│   └── figuras/                # Imagens .png das matrizes de atenção (geradas ao rodar)
├── glossario-capitulo-3.md     # Glossário técnico do Capítulo 3
└── ANALISE_RESULTADOS.md       # Análise dos resultados dos experimentos
```

## Como executar

Todos os arquivos desta Sprint dependem de PyTorch — rode na sua
máquina local, com o ambiente já configurado. O script de experimentos
também usa `matplotlib` para gerar as visualizações.

```bash
# Instalar matplotlib, caso ainda não tenha:
pip install matplotlib

# Ver as implementações funcionando (demonstração rápida):
python src/attention.py

# Rodar todos os experimentos e gerar as visualizações:
python experiments/run_experiments.py
```

As imagens das matrizes de atenção serão salvas em
`experiments/figuras/`.

## O que cada implementação faz

| Mecanismo | Classe/Função | Descrição |
|---|---|---|
| Self-Attention (simples) | `simple_self_attention()` | Versão didática, sem pesos aprendíveis |
| Self-Attention (com pesos) | `SelfAttentionV1` | Projeções Q, K, V aprendíveis |
| Scaled Dot-Product | `scaled_dot_product_attention()` | Função isolada e reutilizável |
| Causal Attention | `CausalAttention` | Self-Attention + máscara causal + dropout, com suporte a batch |
| Multi-Head Attention | `MultiHeadAttention` | Múltiplas heads em paralelo + projeção de saída |

## Experimentos realizados

1. Diferentes dimensões de embedding
2. Diferentes números de heads
3. Diferentes dimensões de head
4. Comparação Self-Attention vs Multi-Head Attention
5. Visualização da matriz de atenção (heatmap)
6. Comparação Attention com e sem escala
7. Comportamento da máscara causal
8. Diferentes sequências de entrada

Resultados e discussão completos em
[`ANALISE_RESULTADOS.md`](./ANALISE_RESULTADOS.md).

## Observação sobre validação

A lógica matemática de todos os mecanismos (fórmulas, formas dos
tensores, comportamento da máscara) foi validada com NumPy antes da
entrega, garantindo que a implementação em PyTorch está correta.
Recomenda-se rodar `src/attention.py` primeiro para confirmar que
tudo funciona no seu ambiente antes de rodar os experimentos completos.
