"""
run_experiments.py

Experimentos da Sprint 4. Cada experimento altera UM parametro por vez
(os demais ficam no valor base) e registra configuracao, parametros,
tempo de forward e forma dos logits.

Ao final, salva os resultados reais da SUA maquina em
RESULTADOS_EXPERIMENTOS.md (nesta pasta), para uso no relatorio.

Observacao: o modelo NAO e treinado nesta sprint; os pesos sao
aleatorios. Logo, o texto gerado nao tem sentido - o que se observa
e o comportamento mecanico da arquitetura e da geracao.
"""

import os
import sys
import time

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from gpt import GPTModel, generate, count_parameters  # noqa: E402

OUT_FILE = os.path.join(os.path.dirname(__file__), "RESULTADOS_EXPERIMENTOS.md")

BASE = {
    "vocab_size": 1000,
    "context_length": 64,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "ff_dim": None,  # None -> 4 * emb_dim
}
BATCH, SEQ = 4, 32

linhas_md = []


def log(txt=""):
    print(txt)
    linhas_md.append(txt)


def medir(cfg, repeticoes=5):
    """Cria o modelo, mede parametros e tempo medio de forward (eval)."""
    torch.manual_seed(123)
    model = GPTModel(cfg).eval()
    x = torch.randint(0, cfg["vocab_size"], (BATCH, min(SEQ, cfg["context_length"])))
    with torch.no_grad():
        model(x)  # aquecimento
        t0 = time.perf_counter()
        for _ in range(repeticoes):
            logits = model(x)
        ms = (time.perf_counter() - t0) / repeticoes * 1000
    return count_parameters(model), ms, tuple(logits.shape)


def experimento(titulo, parametro, valores, obs):
    log("=" * 70)
    log(f"## {titulo}")
    log("=" * 70)
    log(f"Base: {{{', '.join(f'{k}={v}' for k, v in BASE.items())}}}")
    log(f"Entrada: batch={BATCH}, seq={SEQ}\n")
    log(f"| {parametro} | parametros | forward (ms) | logits (B,T,V) |")
    log("|---|---|---|---|")
    for v in valores:
        cfg = dict(BASE)
        cfg[parametro] = v
        n, ms, shape = medir(cfg)
        log(f"| {v} | {n:,} | {ms:.2f} | {shape} |")
    log(f"\nObservacao: {obs}\n")


def exp_geracao():
    log("=" * 70)
    log("## Experimento 7 - Parametros de geracao (temperatura e top-k)")
    log("=" * 70)
    torch.manual_seed(123)
    cfg = dict(BASE)
    model = GPTModel(cfg).eval()
    prompt = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        logits = model(prompt)[:, -1, :]

    log("Entropia da distribuicao do proximo token (mais alta = mais aleatoria)")
    log("| temperatura | entropia (nats) | prob. maxima |")
    log("|---|---|---|")
    for temp in [0.2, 0.5, 1.0, 2.0, 5.0]:
        p = torch.softmax(logits / temp, dim=-1)
        ent = -(p * torch.log(p + 1e-12)).sum().item()
        log(f"| {temp} | {ent:.3f} | {p.max().item():.4f} |")

    log("\nDiversidade na geracao: 10 amostras de 20 tokens cada")
    log("| temperatura | top_k | tokens distintos (media) |")
    log("|---|---|---|")
    configs = [(0.0, None), (1.0, None), (1.0, 50), (1.0, 5), (1.0, 2), (2.0, None)]
    for temp, k in configs:
        distintos = []
        for s in range(10):
            torch.manual_seed(s)
            out = generate(model, prompt, 20, cfg["context_length"],
                           temperature=temp, top_k=k)
            distintos.append(len(set(out[0, 4:].tolist())))
        log(f"| {temp} | {k} | {sum(distintos) / len(distintos):.1f} |")

    # greedy e deterministico
    a = generate(model, prompt, 10, cfg["context_length"], temperature=0.0)
    b = generate(model, prompt, 10, cfg["context_length"], temperature=0.0)
    log(f"\nGreedy deterministico (duas execucoes iguais): {bool((a == b).all())}")
    log("\nObservacao: temperatura baixa concentra a probabilidade em poucos")
    log("tokens (entropia menor); temperatura alta achata a distribuicao.")
    log("top-k corta a cauda: com k pequeno, so os k tokens mais provaveis")
    log("podem ser sorteados, reduzindo a diversidade.\n")


def exp_residual_layernorm():
    log("=" * 70)
    log("## Experimento 8 - Efeito da LayerNorm e das conexoes residuais")
    log("=" * 70)
    torch.manual_seed(123)
    from gpt import LayerNorm

    x = torch.randn(2, 5, 128) * 3 + 7
    y = LayerNorm(128)(x)
    log(f"Antes da LayerNorm : media={x.mean().item():.3f}, desvio={x.std().item():.3f}")
    log(f"Depois da LayerNorm: media={y.mean().item():.3f}, desvio={y.std().item():.3f}")

    # Sensibilidade a profundidade: variancia da saida com pilhas diferentes
    log("\nDesvio padrao da saida (logits) conforme o numero de blocos:")
    log("| n_layers | desvio dos logits |")
    log("|---|---|")
    for n in [1, 2, 4, 8, 12]:
        cfg = dict(BASE)
        cfg["n_layers"] = n
        torch.manual_seed(123)
        model = GPTModel(cfg).eval()
        xi = torch.randint(0, cfg["vocab_size"], (BATCH, SEQ))
        with torch.no_grad():
            lg = model(xi)
        log(f"| {n} | {lg.std().item():.4f} |")
    log("\nObservacao: gracas a LayerNorm e as conexoes residuais, a escala")
    log("da saida permanece estavel mesmo com pilhas mais profundas.\n")


def exp_dropout():
    log("=" * 70)
    log("## Experimento 9 - Dropout (modo treino vs modo inferencia)")
    log("=" * 70)
    x = torch.randint(0, BASE["vocab_size"], (BATCH, SEQ))
    log("| drop_rate | modo | diferenca entre 2 passagens (max abs) |")
    log("|---|---|---|")
    for rate in [0.0, 0.1, 0.3, 0.5]:
        cfg = dict(BASE)
        cfg["drop_rate"] = rate
        torch.manual_seed(123)
        model = GPTModel(cfg)
        for modo in ["train", "eval"]:
            getattr(model, modo)()
            with torch.no_grad():
                a, b = model(x), model(x)
            log(f"| {rate} | {modo} | {(a - b).abs().max().item():.5f} |")
    log("\nObservacao: em modo treino o dropout zera ativacoes aleatoriamente,")
    log("entao duas passagens identicas divergem; em modo inferencia (eval) o")
    log("dropout e desativado e o resultado e deterministico. Dropout nao altera")
    log("o numero de parametros nem o custo de forma relevante.\n")


if __name__ == "__main__":
    log("# Resultados dos Experimentos - Sprint 4\n")
    experimento("Experimento 1 - Numero de Transformer Blocks", "n_layers",
                [1, 2, 4, 8, 12],
                "parametros e tempo crescem aproximadamente de forma linear "
                "com o numero de blocos (cada bloco tem o mesmo custo).")
    experimento("Experimento 2 - Numero de heads", "n_heads",
                [1, 2, 4, 8, 16],
                "o numero de parametros NAO muda (as matrizes Wq/Wk/Wv continuam "
                "D x D); muda apenas como a dimensao D e dividida entre as heads.")
    experimento("Experimento 3 - Dimensao dos embeddings", "emb_dim",
                [32, 64, 128, 256],
                "os parametros crescem de forma aproximadamente quadratica em "
                "emb_dim (blocos) mais linear (embeddings e cabeca de saida).")
    experimento("Experimento 4 - Dimensao da FeedForward", "ff_dim",
                [128, 256, 512, 1024],
                "a FeedForward concentra a maior parte dos parametros de cada "
                "bloco; aumentar ff_dim impacta diretamente o custo.")
    experimento("Experimento 5 - Tamanho do contexto", "context_length",
                [16, 32, 64, 128],
                "aumenta apenas a tabela de embeddings posicionais (e a mascara); "
                "o tempo de forward depende de SEQ, nao de context_length.")
    experimento("Experimento 6 - Tamanho do vocabulario", "vocab_size",
                [500, 1000, 5000, 20000],
                "vocab_size afeta a tabela de embeddings e a cabeca de saida "
                "(D x V): em vocabularios grandes, essas camadas dominam o modelo.")
    exp_geracao()
    exp_residual_layernorm()
    exp_dropout()

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas_md))
    print(f"\nResultados salvos em: {OUT_FILE}")
