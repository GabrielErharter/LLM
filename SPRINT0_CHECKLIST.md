# Sprint 0 — Checklist, Dependências e Riscos

## Fases

### Fase 1 — Repositório
- [ ] Criar repositório (GitHub/GitLab)
- [ ] Criar estrutura de diretórios (src/, notebooks/, docs/, glossario/, experimentos/, relatorios/)
- [ ] Definir convenção de commits/branches

### Fase 2 — Ambiente de Desenvolvimento
- [ ] Instalar Python + PyTorch
- [ ] Gerar `requirements.txt`
- [ ] Validar execução em **ambos** os computadores da dupla
- [ ] Rodar script de validação (tensor básico)

### Fase 3 — Leituras (podem ser paralelas entre si)
- [ ] 3a. Ler Capítulo 1 — Understanding Large Language Models
  - [ ] O que caracteriza um LLM?
  - [ ] Diferença entre IA, ML e Deep Learning?
  - [ ] Como os modelos GPT estão organizados?
  - [ ] Principais etapas do treinamento de um LLM?
  - [ ] Fluxo geral de funcionamento de um modelo de linguagem?
- [ ] 3b. Ler Apêndice sobre PyTorch
  - [ ] Criação de tensores
  - [ ] Operações básicas
  - [ ] Manipulação de dimensões
  - [ ] Operações matriciais
  - [ ] Autograd
  - [ ] Construção de modelos com módulos do PyTorch

### Fase 4 — Glossário
- [ ] Elaborar glossário do Capítulo 1 (`glossario/capitulo-1.md`)

### Fase 5 — Consolidação e Entrega
- [ ] Preencher README.md com descrição do projeto
- [ ] Conferir ambiente funcional para os dois integrantes
- [ ] Push final
- [ ] Enviar link do repositório

## Dependências

```
Fase 1 (Repositório)
   │
   ▼
Fase 2 (Ambiente)
   │
   ▼
Fase 3 (Leituras 3a e 3b — paralelas)
   │
   ▼
Fase 4 (Glossário) ── depende diretamente de 3a
   │
   ▼
Fase 5 (Entrega) ── depende de todas as anteriores
```

O **ambiente (Fase 2)** é o ponto crítico: sem ele validado nos dois computadores, a entrega final trava.

## Riscos e Mitigações

| Risco | Impacto | Mitigação |
|---|---|---|
| Ambiente PyTorch divergente entre integrantes (versões de Python/SO diferentes) | Alto — "funciona só numa máquina" | Fixar versões em `requirements.txt` desde o início |
| Leitura superficial do Capítulo 1 | Médio — glossário raso, prejudica sprints futuras | Usar as perguntas-guia como checklist antes de escrever o glossário |
| Divisão de trabalho desbalanceada na dupla | Médio — um integrante fica sem base teórica | Alternar quem lê e quem configura; revisar tudo em dupla |
| Estrutura de diretórios improvisada | Baixo/Médio — retrabalho nas próximas sprints | Definir estrutura completa já na Fase 1, mesmo com pastas vazias |
| Prazo subestimado (configuração de PyTorch/GPU pode travar) | Alto | Começar pelo ambiente no primeiro dia, não deixar para o fim
