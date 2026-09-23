"""
attention.py

Implementa, de forma incremental, os mecanismos de atenção do Capítulo 3:

1. SelfAttentionSimple      -> versão simples, sem pesos aprendíveis (didática)
2. SelfAttentionV1          -> Self-Attention com Q, K, V aprendíveis
3. ScaledDotProductAttention -> mesma lógica, isolada em função reutilizável
4. CausalAttention           -> Self-Attention + máscara causal + dropout
5. MultiHeadAttention        -> várias CausalAttention em paralelo, combinadas

Execute na máquina local, onde o PyTorch já está instalado.
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# 1. Self-Attention simples (sem pesos aprendíveis) — versão didática
# ---------------------------------------------------------------------
def simple_self_attention(inputs: torch.Tensor) -> torch.Tensor:
    """
    Calcula a atenção de forma simplificada, usando o próprio embedding
    de entrada como Query, Key e Value (sem projeções aprendíveis).

    Args:
        inputs: tensor (context_size, embedding_dim)

    Returns:
        context_vectors: tensor (context_size, embedding_dim)
    """
    # Scores brutos: produto escalar entre cada par de tokens
    attn_scores = inputs @ inputs.T  # (context_size, context_size)

    # Normaliza os scores em pesos de atenção (soma 1 por linha)
    attn_weights = torch.softmax(attn_scores, dim=-1)

    # Combina os valores (aqui, os próprios inputs) ponderados pelos pesos
    context_vectors = attn_weights @ inputs

    return context_vectors, attn_weights


# ---------------------------------------------------------------------
# 2 e 3. Self-Attention com Q, K, V aprendíveis + Scaled Dot-Product
# ---------------------------------------------------------------------
class SelfAttentionV1(nn.Module):
    """
    Self-Attention com matrizes de projeção Wq, Wk, Wv aprendíveis,
    aplicando o Scaled Dot-Product Attention.
    """

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (context_size, d_in)
        Returns:
            context_vec: tensor (context_size, d_out)
        """
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # ---- Scaled Dot-Product Attention ----
        attn_scores = queries @ keys.T  # (context_size, context_size)
        attn_scores_scaled = attn_scores / math.sqrt(self.d_out)
        attn_weights = torch.softmax(attn_scores_scaled, dim=-1)

        context_vec = attn_weights @ values
        return context_vec, attn_weights


def scaled_dot_product_attention(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Função isolada que implementa o Scaled Dot-Product Attention,
    reutilizável por qualquer camada (CausalAttention, MultiHeadAttention).

    Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d_k)) V
    """
    d_k = keys.shape[-1]
    attn_scores = queries @ keys.transpose(-2, -1)
    attn_scores_scaled = attn_scores / math.sqrt(d_k)
    attn_weights = torch.softmax(attn_scores_scaled, dim=-1)
    context_vec = attn_weights @ values
    return context_vec, attn_weights


# ---------------------------------------------------------------------
# 4. Causal Attention (com máscara + dropout)
# ---------------------------------------------------------------------
class CausalAttention(nn.Module):
    """
    Self-Attention com máscara causal (cada token só vê tokens anteriores
    a ele, incluindo ele mesmo) e dropout aplicado aos pesos de atenção.
    Suporta lotes (batch_size, context_size, d_in).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # Máscara triangular superior (1s acima da diagonal = posições futuras)
        # registrada como buffer para ser movida automaticamente para GPU/CPU
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (batch_size, context_size, d_in)
        Returns:
            context_vec: tensor (batch_size, context_size, d_out)
        """
        batch_size, context_size, _ = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)  # (batch, context, context)

        # Aplica a máscara causal: posições futuras recebem -infinito
        attn_scores = attn_scores.masked_fill(
            self.mask.bool()[:context_size, :context_size], float("-inf")
        )

        attn_weights = torch.softmax(attn_scores / math.sqrt(self.d_out), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec, attn_weights


# ---------------------------------------------------------------------
# 5. Multi-Head Attention
# ---------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    Executa múltiplas 'heads' de Causal Attention em paralelo e combina
    os resultados através de uma projeção linear final.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out precisa ser divisível por num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)  # projeção final após concatenar as heads
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (batch_size, context_size, d_in)
        Returns:
            context_vec: tensor (batch_size, context_size, d_out)
        """
        batch_size, context_size, _ = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Divide d_out em (num_heads, head_dim) e reorganiza as dimensões
        # para (batch, num_heads, context, head_dim)
        queries = queries.view(batch_size, context_size, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, context_size, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, context_size, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention, agora em paralelo para todas as heads
        attn_scores = queries @ keys.transpose(2, 3)  # (batch, heads, context, context)
        attn_scores = attn_scores.masked_fill(
            self.mask.bool()[:context_size, :context_size], float("-inf")
        )
        attn_weights = torch.softmax(attn_scores / math.sqrt(self.head_dim), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values  # (batch, heads, context, head_dim)

        # Reorganiza de volta para (batch, context, d_out) concatenando as heads
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, context_size, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec, attn_weights


if __name__ == "__main__":
    torch.manual_seed(123)

    # Entrada de exemplo: 1 sequência de 4 tokens, embedding_dim=8
    inputs = torch.rand(4, 8)

    print("### 1. Self-Attention simples (sem pesos aprendíveis) ###")
    context, weights = simple_self_attention(inputs)
    print(f"Forma dos vetores de contexto: {context.shape}")
    print(f"Forma da matriz de atenção   : {weights.shape}")
    print(f"Soma de cada linha da matriz de atenção (deve ser ~1): {weights.sum(dim=-1)}")
    print()

    print("### 2/3. Self-Attention com Q,K,V aprendíveis (Scaled Dot-Product) ###")
    torch.manual_seed(123)
    sa_v1 = SelfAttentionV1(d_in=8, d_out=8)
    context_v1, weights_v1 = sa_v1(inputs)
    print(f"Forma dos vetores de contexto: {context_v1.shape}")
    print()

    print("### 4. Causal Attention ###")
    batch_inputs = torch.stack([inputs, inputs])  # simula batch_size=2
    torch.manual_seed(123)
    causal_attn = CausalAttention(d_in=8, d_out=8, context_length=4, dropout=0.0)
    context_causal, weights_causal = causal_attn(batch_inputs)
    print(f"Forma dos vetores de contexto: {context_causal.shape}")
    print(f"Matriz de atenção (primeira amostra do batch):\n{weights_causal[0]}")
    print()

    print("### 5. Multi-Head Attention ###")
    torch.manual_seed(123)
    mha = MultiHeadAttention(d_in=8, d_out=8, context_length=4, num_heads=2, dropout=0.0)
    context_mha, weights_mha = mha(batch_inputs)
    print(f"Forma dos vetores de contexto: {context_mha.shape}")
    print(f"Forma da matriz de atenção (todas as heads): {weights_mha.shape}")
