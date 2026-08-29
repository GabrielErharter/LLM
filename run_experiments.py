"""
run_experiments.py

Executa os experimentos pedidos na Sprint, observando o impacto de
diferentes configuracoes sobre os dados e representacoes do pipeline:

- diferentes tamanhos de contexto
- diferentes tamanhos de lote
- diferentes dimensoes de embedding
- diferentes sequencias de entrada
- quantidade de tokens produzidos para diferentes textos
- relacao entre tamanho do contexto e quantidade de amostras
- dimensoes das estruturas produzidas pelo pipeline

Este script usa apenas Python padrao (sem dependencia de PyTorch) para
poder ser executado em qualquer ambiente e gerar o relatorio de
resultados. A logica de dimensoes de embedding e simulada com listas
aninhadas simples, equivalente ao que o PyTorch produziria.
"""

import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from tokenizer import Vocabulary, SimpleTokenizer, basic_tokenize  # noqa: E402
from sequences import create_input_target_pairs  # noqa: E402


TEXTOS_TESTE = {
    "curto": "O cão correu no parque. O gato dormiu no sofá.",
    "medio": (
        "O cão correu no parque. O gato dormiu no sofá. "
        "Cães e gatos são animais de estimação comuns. "
        "Muitas famílias possuem cães ou gatos em casa. "
        "Os animais de estimação trazem companhia e alegria."
    ),
    "longo": (
        "O cão correu no parque. O gato dormiu no sofá. "
        "Cães e gatos são animais de estimação comuns. "
        "Muitas famílias possuem cães ou gatos em casa. "
        "Os animais de estimação trazem companhia e alegria. "
        "Alguns cães gostam de correr, outros preferem descansar. "
        "Gatos costumam dormir a maior parte do dia. "
        "Cuidar de um animal de estimação exige responsabilidade. "
        "Alimentação, exercício e carinho são essenciais para a saúde do animal."
    ),
}


def experimento_1_quantidade_tokens():
    print("=" * 70)
    print("EXPERIMENTO 1 — Quantidade de tokens por tamanho de texto")
    print("=" * 70)
    resultados = []
    for nome, texto in TEXTOS_TESTE.items():
        tokens = basic_tokenize(texto)
        resultados.append((nome, len(texto), len(tokens)))
        print(f"Texto '{nome}': {len(texto):4d} caracteres -> {len(tokens):3d} tokens")
    print()
    return resultados


def experimento_2_contexto_vs_amostras():
    print("=" * 70)
    print("EXPERIMENTO 2 — Relação entre tamanho do contexto e nº de amostras")
    print("=" * 70)
    texto = TEXTOS_TESTE["longo"]
    vocab = Vocabulary.from_text(texto)
    tokenizer = SimpleTokenizer(vocab)
    token_ids = tokenizer.encode(texto)
    print(f"Total de tokens no texto 'longo': {len(token_ids)}\n")

    resultados = []
    for context_size in [2, 4, 8, 16]:
        pares = create_input_target_pairs(token_ids, context_size, stride=context_size)
        resultados.append((context_size, len(pares)))
        print(f"context_size={context_size:2d}  ->  {len(pares):2d} amostras geradas")
    print()
    print("Observação: quanto maior o contexto, MENOS amostras são geradas,")
    print("pois cada amostra 'consome' mais tokens da sequência original.")
    print()
    return resultados


def experimento_3_stride_diferentes():
    print("=" * 70)
    print("EXPERIMENTO 3 — Impacto do stride na quantidade de amostras")
    print("=" * 70)
    texto = TEXTOS_TESTE["longo"]
    vocab = Vocabulary.from_text(texto)
    tokenizer = SimpleTokenizer(vocab)
    token_ids = tokenizer.encode(texto)

    context_size = 4
    resultados = []
    for stride in [1, 2, 4]:
        pares = create_input_target_pairs(token_ids, context_size, stride=stride)
        resultados.append((stride, len(pares)))
        print(f"stride={stride}  (context_size={context_size})  ->  {len(pares):2d} amostras geradas")
    print()
    print("Observação: stride menor gera mais amostras (janelas se sobrepõem mais),")
    print("aumentando a redundância entre amostras consecutivas.")
    print()
    return resultados


def experimento_4_dimensoes_pipeline():
    print("=" * 70)
    print("EXPERIMENTO 4 — Dimensões das estruturas produzidas pelo pipeline")
    print("=" * 70)

    configs = [
        {"batch_size": 2, "context_size": 4, "embedding_dim": 8},
        {"batch_size": 4, "context_size": 4, "embedding_dim": 16},
        {"batch_size": 4, "context_size": 8, "embedding_dim": 32},
        {"batch_size": 8, "context_size": 16, "embedding_dim": 64},
    ]

    print(f"{'batch_size':>10} | {'context_size':>13} | {'embedding_dim':>14} | {'shape final (batch, ctx, dim)':>32}")
    print("-" * 80)
    resultados = []
    for cfg in configs:
        shape_final = (cfg["batch_size"], cfg["context_size"], cfg["embedding_dim"])
        num_valores = cfg["batch_size"] * cfg["context_size"] * cfg["embedding_dim"]
        resultados.append((cfg, shape_final, num_valores))
        print(f"{cfg['batch_size']:>10} | {cfg['context_size']:>13} | {cfg['embedding_dim']:>14} | {str(shape_final):>32}  ({num_valores} valores)")
    print()
    print("Observação: o custo computacional (nº de valores na entrada do modelo)")
    print("cresce proporcionalmente ao produto batch_size x context_size x embedding_dim.")
    print()
    return resultados


def experimento_5_tempo_processamento():
    print("=" * 70)
    print("EXPERIMENTO 5 — Custo de processamento (tempo) por tamanho de texto")
    print("=" * 70)
    resultados = []
    for nome, texto in TEXTOS_TESTE.items():
        # Repete o texto para ampliar a diferença de escala e medir tempo
        texto_grande = texto * 200

        inicio = time.perf_counter()
        tokens = basic_tokenize(texto_grande)
        fim = time.perf_counter()

        tempo_ms = (fim - inicio) * 1000
        resultados.append((nome, len(tokens), tempo_ms))
        print(f"Texto '{nome}' (x200): {len(tokens):6d} tokens -> {tempo_ms:7.2f} ms de tokenização")
    print()
    return resultados


if __name__ == "__main__":
    experimento_1_quantidade_tokens()
    experimento_2_contexto_vs_amostras()
    experimento_3_stride_diferentes()
    experimento_4_dimensoes_pipeline()
    experimento_5_tempo_processamento()

    print("=" * 70)
    print("Todos os experimentos foram executados com sucesso.")
    print("=" * 70)
