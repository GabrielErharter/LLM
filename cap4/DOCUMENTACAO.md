# Documentação — Transformer Block, GPT e Geração

Notação: **B** = batch, **T** = tokens na sequência, **D** = `emb_dim`,
**H** = `n_heads`, **d_h** = D/H, **F** = dimensão da FeedForward (padrão 4D),
**V** = `vocab_size`.

## 1. Transformer Block (`TransformerBlock` em `src/gpt.py`)

```
x (B,T,D)
 ├─► LayerNorm ─► Multi-Head Attention ─► Dropout ─► (+) ──┐   sub-bloco 1
 └─────────────────────────────────────────────────────────┘   (residual)
 x (B,T,D)
 ├─► LayerNorm ─► FeedForward ─► Dropout ─► (+) ───────────┐   sub-bloco 2
 └─────────────────────────────────────────────────────────┘   (residual)
 saída (B,T,D)
```

| Etapa | Tensor de entrada | Tensor de saída | Função |
|---|---|---|---|
| LayerNorm 1 | (B,T,D) | (B,T,D) | Normaliza cada token (média 0, variância 1) |
| Projeções Q,K,V | (B,T,D) | 3 × (B,T,D) → (B,H,T,d_h) | Divide D entre as H heads |
| Scores QKᵀ | (B,H,T,d_h) | (B,H,T,T) | Similaridade entre tokens |
| Máscara + Softmax | (B,H,T,T) | (B,H,T,T) | Bloqueia o futuro; pesos somam 1 |
| Pesos × V | (B,H,T,T) | (B,H,T,d_h) → (B,T,D) | Combina valores e concatena as heads |
| `out_proj` | (B,T,D) | (B,T,D) | Mistura a informação das heads |
| Soma residual 1 | (B,T,D) + (B,T,D) | (B,T,D) | Atalho para o gradiente |
| LayerNorm 2 | (B,T,D) | (B,T,D) | Normaliza antes da FFN |
| FFN: Linear D→F | (B,T,D) | (B,T,F) | Expande a dimensão |
| GELU | (B,T,F) | (B,T,F) | Não linearidade |
| FFN: Linear F→D | (B,T,F) | (B,T,D) | Volta à dimensão D |
| Soma residual 2 | (B,T,D) + (B,T,D) | (B,T,D) | Atalho para o gradiente |

Como entrada e saída têm a mesma forma, os blocos podem ser empilhados.

## 2. Modelo GPT (`GPTModel`)

```
Token IDs (B,T)
  → tok_emb + pos_emb ............ (B,T,D)
  → Dropout ...................... (B,T,D)
  → N × TransformerBlock ......... (B,T,D)
  → LayerNorm final .............. (B,T,D)
  → out_head (Linear D→V) ........ logits (B,T,V)
```

Mapeamento no código: `self.tok_emb`, `self.pos_emb` (Sprint 1),
`self.trf_blocks` (N blocos, atenção da Sprint 3), `self.final_norm`,
`self.out_head` (Language Model Head).

## 3. Inferência e geração autorregressiva (`generate`)

Procedimento, repetido `max_new_tokens` vezes:

1. **Recortar o contexto** aos últimos `context_length` tokens: `idx[:, -context_size:]`.
2. **Forward** do modelo: logits `(B,T,V)`; usa-se só a última posição → `(B,V)`.
3. **Top-k (opcional):** mantém os k maiores logits; os demais viram `-∞`.
4. **Probabilidades:** `softmax(logits / temperatura)` (temperatura > 0).
5. **Seleção do próximo token:** `argmax` (greedy, temperatura = 0) ou
   `multinomial` (amostragem).
6. **Concatenar** o token escolhido ao contexto e voltar ao passo 1.

Relação entre os conceitos: **logits** são pontuações brutas → o **softmax**
as converte em **probabilidades** → a **estratégia de seleção** escolhe um
token → ele entra no contexto, e o ciclo se repete (**autorregressão**).
A **máscara causal** garante que, no treinamento, cada posição só dependa do
passado, o que torna esse uso na inferência consistente.

> Nesta sprint o modelo **não foi treinado** (pesos aleatórios). O texto
> gerado não tem sentido; o que se demonstra é o mecanismo de geração.
