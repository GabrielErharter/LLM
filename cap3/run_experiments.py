"""
run_experiments.py

Executa os experimentos pedidos na Sprint 3, observando o comportamento
do mecanismo de atenção sob diferentes configurações:

- diferentes dimensões de embedding
- diferentes números de heads
- diferentes dimensões de head
- comparação entre Self-Attention e Multi-Head Attention
- visualização da matriz de atenção
- comparação entre Attention com e sem escala
- comportamento da máscara causal
- diferentes sequências de entrada

Execute na máquina local, onde o PyTorch já está instalado.
Gera imagens .png das matrizes de atenção na pasta 'figuras/'.
"""

import os
import sys
import math

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from attention import (  # noqa: E402
    simple_self_attention,
    SelfAttentionV1,
    CausalAttention,
    MultiHeadAttention,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figuras")
os.makedirs(FIG_DIR, exist_ok=True)


def salvar_matriz_atencao(weights: torch.Tensor, titulo: str, filename: str):
    """Salva uma imagem (heatmap) da matriz de atenção."""
    plt.figure(figsize=(5, 4))
    plt.imshow(weights.detach().numpy(), cmap="viridis")
    plt.colorbar(label="Peso de atenção")
    plt.title(titulo)
    plt.xlabel("Posição (Key)")
    plt.ylabel("Posição (Query)")
    plt.tight_layout()
    caminho = os.path.join(FIG_DIR, filename)
    plt.savefig(caminho, dpi=120)
    plt.close()
    print(f"  Figura salva em: {caminho}")


def experimento_1_dimensoes_embedding():
    print("=" * 70)
    print("EXPERIMENTO 1 — Diferentes dimensões de embedding")
    print("=" * 70)
    context_size = 6
    torch.manual_seed(123)

    for d_model in [8, 16, 32, 64]:
        x = torch.rand(1, context_size, d_model)
        attn = CausalAttention(d_in=d_model, d_out=d_model, context_length=context_size)
        out, weights = attn(x)
        print(f"d_model={d_model:3d}  ->  saída: {tuple(out.shape)}  |  matriz de atenção: {tuple(weights.shape)}")
    print()


def experimento_2_numero_de_heads():
    print("=" * 70)
    print("EXPERIMENTO 2 — Diferentes números de heads (d_model fixo = 32)")
    print("=" * 70)
    context_size = 6
    d_model = 32
    torch.manual_seed(123)
    x = torch.rand(1, context_size, d_model)

    for num_heads in [1, 2, 4, 8]:
        mha = MultiHeadAttention(d_in=d_model, d_out=d_model, context_length=context_size, num_heads=num_heads)
        out, weights = mha(x)
        head_dim = d_model // num_heads
        n_params = sum(p.numel() for p in mha.parameters())
        print(f"num_heads={num_heads}  |  head_dim={head_dim:3d}  |  saída: {tuple(out.shape)}  |  attn shape: {tuple(weights.shape)}  |  parâmetros: {n_params}")
    print()


def experimento_3_dimensao_head():
    print("=" * 70)
    print("EXPERIMENTO 3 — Diferentes dimensões de head (variando d_model, heads fixo=4)")
    print("=" * 70)
    context_size = 6
    num_heads = 4
    torch.manual_seed(123)

    for d_model in [16, 32, 64, 128]:
        x = torch.rand(1, context_size, d_model)
        mha = MultiHeadAttention(d_in=d_model, d_out=d_model, context_length=context_size, num_heads=num_heads)
        head_dim = d_model // num_heads
        out, _ = mha(x)
        print(f"d_model={d_model:4d}  |  head_dim={head_dim:3d}  |  saída: {tuple(out.shape)}")
    print()


def experimento_4_self_vs_multihead():
    print("=" * 70)
    print("EXPERIMENTO 4 — Comparação: Self-Attention (1 head) vs Multi-Head")
    print("=" * 70)
    context_size = 6
    d_model = 32
    torch.manual_seed(123)
    x = torch.rand(context_size, d_model)
    x_batch = x.unsqueeze(0)

    torch.manual_seed(123)
    sa = SelfAttentionV1(d_in=d_model, d_out=d_model)
    context_sa, weights_sa = sa(x)

    torch.manual_seed(123)
    mha = MultiHeadAttention(d_in=d_model, d_out=d_model, context_length=context_size, num_heads=4)
    context_mha, weights_mha = mha(x_batch)

    n_params_sa = sum(p.numel() for p in sa.parameters())
    n_params_mha = sum(p.numel() for p in mha.parameters())

    print(f"Self-Attention (1 'head' efetiva): {n_params_sa} parâmetros")
    print(f"Multi-Head Attention (4 heads)    : {n_params_mha} parâmetros")
    print()
    print("Observação: a Self-Attention simples não possui máscara causal nem")
    print("múltiplas heads — captura apenas UM tipo de relação entre os tokens.")
    print("O Multi-Head Attention, com o mesmo d_model, aprende múltiplas")
    print("'perspectivas' de relação em paralelo, com custo extra de parâmetros")
    print("apenas na projeção de saída (out_proj).")
    print()

    salvar_matriz_atencao(weights_sa, "Self-Attention (sem máscara)", "exp4_self_attention.png")
    print()
    return weights_sa, weights_mha


def experimento_5_visualizacao_matriz():
    print("=" * 70)
    print("EXPERIMENTO 5 — Visualização da matriz de atenção (Causal)")
    print("=" * 70)
    context_size = 8
    d_model = 16
    torch.manual_seed(123)
    x = torch.rand(1, context_size, d_model)

    attn = CausalAttention(d_in=d_model, d_out=d_model, context_length=context_size)
    _, weights = attn(x)

    salvar_matriz_atencao(
        weights[0], "Matriz de Atenção Causal (context_size=8)", "exp5_causal_attention.png"
    )
    print("Padrão esperado: triangular inferior — cada token só atende")
    print("a si mesmo e aos tokens anteriores.")
    print()


def experimento_6_com_sem_escala():
    print("=" * 70)
    print("EXPERIMENTO 6 — Attention com e sem escala (scaled vs unscaled)")
    print("=" * 70)
    torch.manual_seed(123)
    context_size = 6
    d_model = 64  # dimensão alta para evidenciar o efeito da escala

    x = torch.rand(context_size, d_model)
    W_q = torch.nn.Linear(d_model, d_model, bias=False)
    W_k = torch.nn.Linear(d_model, d_model, bias=False)

    Q = W_q(x)
    K = W_k(x)

    scores = Q @ K.T

    # Sem escala
    weights_unscaled = torch.softmax(scores, dim=-1)
    # Com escala
    weights_scaled = torch.softmax(scores / math.sqrt(d_model), dim=-1)

    print(f"Desvio padrão dos scores brutos           : {scores.std().item():.4f}")
    print(f"Entropia média (sem escala) — quanto MENOR, mais 'dura' a distribuição:")
    entropy_unscaled = -(weights_unscaled * torch.log(weights_unscaled + 1e-9)).sum(dim=-1).mean()
    entropy_scaled = -(weights_scaled * torch.log(weights_scaled + 1e-9)).sum(dim=-1).mean()
    print(f"  Sem escala : {entropy_unscaled.item():.4f}")
    print(f"  Com escala : {entropy_scaled.item():.4f}")
    print()
    print("Observação: sem o fator de escala (1/sqrt(d_k)), os scores tendem a")
    print("ter variância maior em dimensões altas, levando o Softmax a distribuições")
    print("mais 'extremas' (menor entropia) — próximas de um one-hot, o que satura")
    print("os gradientes durante o treinamento. A escala suaviza a distribuição.")
    print()

    salvar_matriz_atencao(weights_unscaled, "Attention SEM escala", "exp6_sem_escala.png")
    salvar_matriz_atencao(weights_scaled, "Attention COM escala", "exp6_com_escala.png")
    print()


def experimento_7_mascara_causal():
    print("=" * 70)
    print("EXPERIMENTO 7 — Comportamento da máscara causal")
    print("=" * 70)
    context_size = 5
    d_model = 8
    torch.manual_seed(123)
    x = torch.rand(1, context_size, d_model)

    attn = CausalAttention(d_in=d_model, d_out=d_model, context_length=context_size)
    _, weights = attn(x)
    w = weights[0]

    print("Matriz de atenção causal (arredondada):")
    print(torch.round(w * 1000) / 1000)
    print()

    # Verifica que a parte superior é exatamente zero
    upper_triangle_sum = torch.triu(w, diagonal=1).sum().item()
    print(f"Soma dos pesos na parte superior (posições futuras): {upper_triangle_sum:.6f}")
    print("(Deve ser ~0.0 — confirma que a máscara está bloqueando o futuro)")
    print()


def experimento_8_diferentes_sequencias():
    print("=" * 70)
    print("EXPERIMENTO 8 — Diferentes sequências de entrada")
    print("=" * 70)
    d_model = 16
    torch.manual_seed(123)
    attn = CausalAttention(d_in=d_model, d_out=d_model, context_length=20)

    for context_size in [3, 6, 12]:
        x = torch.rand(1, context_size, d_model)
        out, weights = attn(x)
        print(f"context_size={context_size:2d}  ->  saída: {tuple(out.shape)}  |  matriz de atenção: {tuple(weights.shape)}")
    print()
    print("Observação: a mesma camada de atenção (mesmos pesos aprendidos)")
    print("se adapta a diferentes tamanhos de sequência, desde que não ultrapassem")
    print("o context_length máximo configurado (20, neste caso).")
    print()


if __name__ == "__main__":
    experimento_1_dimensoes_embedding()
    experimento_2_numero_de_heads()
    experimento_3_dimensao_head()
    experimento_4_self_vs_multihead()
    experimento_5_visualizacao_matriz()
    experimento_6_com_sem_escala()
    experimento_7_mascara_causal()
    experimento_8_diferentes_sequencias()

    print("=" * 70)
    print("Todos os experimentos foram executados com sucesso.")
    print(f"Figuras salvas em: {FIG_DIR}")
    print("=" * 70)
