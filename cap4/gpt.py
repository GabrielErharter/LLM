"""
gpt.py

Arquitetura GPT simplificada (Capítulo 4), integrando o Multi-Head
Attention da Sprint 3:

    LayerNorm -> GELU -> FeedForward -> TransformerBlock -> GPTModel
    + funcoes de geracao autoregressiva (greedy, temperatura, top-k)

Notacao de dimensoes:
    B = batch_size, T = context_size (tokens), D = emb_dim, V = vocab_size
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Multi-Head Attention (mesma logica da Sprint 3, retornando so o contexto)
# ---------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out precisa ser divisivel por num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, t, _ = x.shape
        q = self.W_query(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(2, 3)  # (B, heads, T, T)
        scores = scores.masked_fill(self.mask.bool()[:t, :t], float("-inf"))
        weights = torch.softmax(scores / math.sqrt(self.head_dim), dim=-1)
        weights = self.dropout(weights)

        ctx = (weights @ v).transpose(1, 2).contiguous().view(b, t, self.d_out)
        return self.out_proj(ctx)  # (B, T, D)


# ---------------------------------------------------------------------
# Layer Normalization
# ---------------------------------------------------------------------
class LayerNorm(nn.Module):
    """
    Normaliza cada vetor de token (ultima dimensao) para media 0 e
    variancia 1, depois aplica escala e deslocamento aprendiveis.
    Estabiliza o treinamento de redes profundas.
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# ---------------------------------------------------------------------
# GELU
# ---------------------------------------------------------------------
class GELU(nn.Module):
    """Ativacao GELU (aproximacao usada no GPT-2), mais suave que a ReLU."""

    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
            )
        )


# ---------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------
class FeedForward(nn.Module):
    """
    Rede densa aplicada a cada token independentemente:
    D -> ff_dim (padrao 4*D) -> GELU -> D
    """

    def __init__(self, emb_dim, ff_dim=None):
        super().__init__()
        ff_dim = ff_dim or 4 * emb_dim
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            GELU(),
            nn.Linear(ff_dim, emb_dim),
        )

    def forward(self, x):
        return self.layers(x)  # (B, T, D) -> (B, T, D)


# ---------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """
    Bloco Transformer (pre-LayerNorm, como no GPT-2):

        x -> LayerNorm -> MultiHeadAttention -> Dropout -> (+ x)   [residual 1]
          -> LayerNorm -> FeedForward        -> Dropout -> (+ x)   [residual 2]

    Entrada e saida tem a mesma forma (B, T, D), o que permite empilhar
    varios blocos.
    """

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg["emb_dim"], cfg.get("ff_dim"))
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Sub-bloco de atencao com conexao residual
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Sub-bloco feedforward com conexao residual
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


# ---------------------------------------------------------------------
# Modelo GPT
# ---------------------------------------------------------------------
class GPTModel(nn.Module):
    """
    Token IDs (B, T)
      -> token embedding + positional embedding -> (B, T, D)
      -> dropout
      -> N x TransformerBlock                    -> (B, T, D)
      -> LayerNorm final                          -> (B, T, D)
      -> Language Model Head (Linear D -> V)      -> logits (B, T, V)
    """

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        _, t = in_idx.shape
        tok = self.tok_emb(in_idx)                              # (B, T, D)
        pos = self.pos_emb(torch.arange(t, device=in_idx.device))  # (T, D)
        x = self.drop_emb(tok + pos)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)                               # (B, T, V)
        return logits


# ---------------------------------------------------------------------
# Geracao autoregressiva
# ---------------------------------------------------------------------
@torch.no_grad()
def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    """
    Gera texto token a token.

    Args:
        idx: tensor (B, T) com os Token IDs iniciais (prompt).
        max_new_tokens: quantos tokens novos gerar.
        context_size: janela maxima que o modelo aceita.
        temperature: 0.0 -> greedy (argmax); >0 -> amostragem multinomial.
        top_k: se definido, mantem apenas os k logits mais altos.
        eos_id: se definido, para quando esse token for gerado.

    Fluxo de cada passo:
        logits -> (ultimo token) -> [top-k] -> [temperatura] -> softmax
        -> selecao (argmax ou amostragem) -> concatena ao contexto
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]          # recorta o contexto
        logits = model(idx_cond)[:, -1, :]         # (B, V): so o ultimo token

        if top_k is not None:
            top_vals, _ = torch.topk(logits, top_k)
            min_val = top_vals[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")), logits
            )

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


GPT_CONFIG_TINY = {
    "vocab_size": 50,
    "context_length": 16,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


if __name__ == "__main__":
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_TINY)

    batch = torch.randint(0, GPT_CONFIG_TINY["vocab_size"], (2, 8))
    logits = model(batch)
    print(f"Entrada (B, T)          : {tuple(batch.shape)}")
    print(f"Logits  (B, T, V)       : {tuple(logits.shape)}")
    print(f"Parametros do modelo    : {count_parameters(model):,}")

    prompt = torch.tensor([[1, 2, 3]])
    out = generate(model, prompt, max_new_tokens=5,
                   context_size=GPT_CONFIG_TINY["context_length"])
    print(f"\nPrompt                  : {prompt.tolist()}")
    print(f"Gerado (greedy)         : {out.tolist()}")
