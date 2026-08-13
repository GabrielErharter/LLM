# [Projeto LLM]

> Projeto desenvolvido ao longo do semestre para estudo e implementação de conceitos de Large Language Models (LLMs), com base no livro *Build a Large Language Model (From Scratch)*.

## Integrantes

- Gabriel Erharter — @GabrielErharter
- Nome 2 — @usuario-github

## Descrição do Projeto

O objetivo geral do projeto é estudar e compreender o funcionamento de modelos de linguagem, com foco nos conceitos de Inteligência Artificial, Machine Learning, Deep Learning e Large Language Models (LLMs). A ideia é entender como o texto é processado e como um modelo consegue aprender padrões para gerar respostas.

A motivação do projeto é conhecer, de forma prática, as principais etapas envolvidas na criação e no funcionamento de um modelo de linguagem, desde a tokenização e os embeddings até o processamento por uma arquitetura Transformer.

Ao final do semestre, espera-se construir um LLM simplificado, treinado do zero, capaz de aprender padrões básicos de um conjunto de textos e gerar novos textos a partir deles. O projeto estará diretamente relacionado à disciplina por envolver conceitos de IA, aprendizado de máquina, processamento de dados e implementação de modelos computacionais.

## Estrutura do Repositório

```
.
├── src/                # Código-fonte do projeto
├── notebooks/          # Notebooks de experimentação e prototipagem
├── docs/                # Documentação geral do projeto
├── glossario/           # Glossários técnicos por capítulo/sprint
├── experimentos/        # Registros e resultados de experimentos
├── relatorios/          # Relatórios entregues em cada Sprint
├── requirements.txt      # Dependências do ambiente Python
└── README.md
```

## Ambiente de Desenvolvimento

### Requisitos
- Python 3.10+
- PyTorch (ver versão exata em `requirements.txt`)

### Instalação

```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### Validação do ambiente

Execute o script abaixo para confirmar que o PyTorch está funcional:

```bash
python -c "import torch; print(torch.__version__); print(torch.tensor([1.0, 2.0, 3.0]))"
```

Se o comando imprimir a versão do PyTorch e o tensor sem erros, o ambiente está pronto.

## Sprints

| Sprint | Descrição | Status |
|--------|-----------|--------|
| Sprint 0 | Setup do repositório, ambiente, leitura Cap. 1 e Apêndice PyTorch, glossário | Em andamento |

## Glossário

Os glossários técnicos de cada capítulo estão disponíveis em [`glossario/`](./glossario/).

## Referências

- Raschka, S. *Build a Large Language Model (From Scratch)*.
