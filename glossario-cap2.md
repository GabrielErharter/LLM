# Glossário Técnico — Capítulo 2: Agentes Inteligentes

**Livro:** *Inteligência Artificial: Uma Abordagem Moderna* — Stuart Russell e Peter Norvig, 4ª edição.

Cada termo apresenta: **termo original, tradução/equivalente técnico, definição, função no modelo, relação com outros conceitos, aspecto técnico e exemplo conceitual/computacional.**

---

## 1. Agente (Agent)

* **Termo original:** Agent
* **Tradução/equivalente:** Agente
* **Definição:** Um agente é qualquer entidade que pode ser considerada como capaz de perceber o ambiente por meio de sensores e agir sobre esse ambiente por meio de atuadores.
* **Função no modelo:** O agente é a entidade responsável por observar o estado do ambiente e selecionar ações de acordo com aquilo que percebe.
* **Relação com outros conceitos:** Um agente está sempre associado a um ambiente, sensores, perceptos, ações e atuadores. Seu comportamento é determinado por uma função de agente e implementado por um programa de agente.
* **Aspecto técnico:** O conceito é bastante geral. Um agente pode ser uma pessoa, um robô, um sistema de software, um agente virtual ou até um sistema simples como um termostato.
* **Exemplo conceitual:** Um robô aspirador percebe sujeira e obstáculos por meio de sensores e utiliza motores para se mover e limpar.

---

## 2. Ambiente (Environment)

* **Termo original:** Environment
* **Tradução/equivalente:** Ambiente
* **Definição:** É tudo aquilo que está fora do agente e que pode influenciar seu comportamento ou ser afetado por suas ações.
* **Função no modelo:** É o espaço no qual o agente opera, fornecendo perceptos e recebendo as ações executadas pelo agente.
* **Relação com outros conceitos:** O ambiente determina quais sensores, atuadores e estratégias são adequados para o agente.
* **Aspecto técnico:** A dificuldade de construir um agente racional depende fortemente das características do ambiente.
* **Exemplo conceitual:** Para um táxi autônomo, o ambiente inclui ruas, veículos, pedestres, semáforos, passageiros, clima e outros elementos relevantes.

---

## 3. Sensor

* **Termo original:** Sensor
* **Tradução/equivalente:** Sensor
* **Definição:** Componente utilizado pelo agente para obter informações sobre o ambiente.
* **Função no modelo:** Permite que o agente perceba aspectos do ambiente e receba informações que serão utilizadas na tomada de decisão.
* **Relação com outros conceitos:** Os sensores produzem perceptos, que são utilizados pelo programa de agente para determinar uma ação.
* **Exemplo conceitual:** Uma câmera, um microfone, um radar ou um sensor de temperatura podem funcionar como sensores.

---

## 4. Atuador

* **Termo original:** Actuator
* **Tradução/equivalente:** Atuador
* **Definição:** Componente através do qual o agente executa ações capazes de modificar o ambiente.
* **Função no modelo:** É a interface de saída do agente para o ambiente.
* **Relação com outros conceitos:** Sensores fornecem informações ao agente, enquanto atuadores permitem que o agente produza efeitos no ambiente.
* **Exemplo conceitual:** Os motores de um robô, o volante e os freios de um carro autônomo e um braço robótico são exemplos de atuadores.

---

## 5. Percepto (Percept)

* **Termo original:** Percept
* **Tradução/equivalente:** Percepto / percepção recebida
* **Definição:** Uma informação obtida pelo agente através de seus sensores em determinado momento.
* **Função no modelo:** É a informação imediata utilizada pelo agente para decidir o que fazer.
* **Relação com outros conceitos:** Uma sequência de perceptos forma o histórico de perceptos, que pode ser utilizado por agentes capazes de manter estado interno.
* **Exemplo conceitual:** No mundo do aspirador, um percepto pode ser:

`[A, Dirty]`

indicando que o agente está na posição A e percebe sujeira.

---

## 6. Sequência de Perceptos (Percept Sequence)

* **Termo original:** Percept Sequence
* **Tradução/equivalente:** Sequência de perceptos / histórico de percepções
* **Definição:** Sequência formada por todos os perceptos recebidos pelo agente desde o início de sua operação.
* **Função no modelo:** Permite representar não apenas o que o agente percebe agora, mas também aquilo que percebeu anteriormente.
* **Relação com outros conceitos:** É a entrada conceitual da função de agente e é especialmente importante para agentes que precisam considerar histórico ou memória.
* **Exemplo conceitual:**

`[Percept₁, Percept₂, Percept₃, ...]`

Um agente pode utilizar essa sequência para inferir informações que não estão disponíveis diretamente no percepto atual.

---

## 7. Ação (Action)

* **Termo original:** Action
* **Tradução/equivalente:** Ação
* **Definição:** Comportamento executado pelo agente que pode produzir uma alteração no ambiente.
* **Função no modelo:** Representa a saída escolhida pelo agente em resposta às informações disponíveis.
* **Relação com outros conceitos:** A ação é escolhida pelo programa de agente a partir dos perceptos e, dependendo do tipo de agente, de estado interno, objetivos ou utilidade.
* **Exemplo conceitual:** Um carro autônomo pode escolher ações como acelerar, frear, virar ou manter a velocidade.

---

## 8. Função de Agente (Agent Function)

* **Termo original:** Agent Function
* **Tradução/equivalente:** Função do agente
* **Definição:** Função matemática que associa cada sequência possível de perceptos a uma ação.
* **Função no modelo:** Descreve abstratamente qual ação o agente deveria executar para cada histórico de percepções.
* **Relação com outros conceitos:** A função de agente é uma descrição abstrata, enquanto o programa de agente é sua implementação concreta.
* **Aspecto técnico:** Pode ser representada conceitualmente por:

`f : P* → A`

onde `P*` representa possíveis sequências de perceptos e `A` representa o conjunto de ações.

* **Exemplo conceitual:** Se o agente percebe `[A, Dirty]`, a função pode determinar que a ação correta seja `Suck`.

---

## 9. Programa de Agente (Agent Program)

* **Termo original:** Agent Program
* **Tradução/equivalente:** Programa do agente
* **Definição:** Implementação computacional responsável por produzir ações a partir das percepções recebidas pelo agente.
* **Função no modelo:** Implementa a função de agente em uma arquitetura física ou computacional.
* **Relação com outros conceitos:** Enquanto a função de agente descreve o comportamento desejado de maneira abstrata, o programa de agente representa como esse comportamento é efetivamente implementado.
* **Exemplo conceitual:**

```text
percepto → programa do agente → ação
```

Um programa simples pode conter regras do tipo:

```text
SE houver sujeira → aspirar
SENÃO SE estiver em A → mover para B
```

---

## 10. Racionalidade (Rationality)

* **Termo original:** Rationality
* **Tradução/equivalente:** Racionalidade
* **Definição:** Capacidade de escolher ações que maximizem o desempenho esperado do agente, considerando as informações disponíveis e as limitações existentes.
* **Função no modelo:** Fornece um critério para determinar se o comportamento de um agente é adequado.
* **Relação com outros conceitos:** Está diretamente relacionada ao agente racional, à medida de desempenho, à sequência de perceptos, ao conhecimento prévio e às ações disponíveis.
* **Aspecto técnico:** Racionalidade não significa que o agente sempre toma a decisão perfeita. O agente deve escolher a melhor ação que pode justificar com base nas informações que possui.
* **Exemplo conceitual:** Se um carro autônomo não sabe que existe um acidente atrás de uma curva porque não possui nenhuma informação sobre ele, não é irracional deixar de evitá-lo.

---

## 11. Agente Racional (Rational Agent)

* **Termo original:** Rational Agent
* **Tradução/equivalente:** Agente racional
* **Definição:** Agente que, para cada sequência de perceptos possível, escolhe a ação que maximiza sua medida de desempenho esperada, considerando as evidências disponíveis.
* **Função no modelo:** É o conceito central utilizado pelo livro para definir comportamento inteligente.
* **Relação com outros conceitos:** Depende da medida de desempenho, dos perceptos, do conhecimento prévio, das ações disponíveis e das características do ambiente.
* **Aspecto técnico:** Um agente racional não precisa ser onisciente. Ele toma decisões utilizando aquilo que sabe ou consegue inferir.
* **Exemplo conceitual:** Um aspirador racional não precisa limpar perfeitamente o ambiente em todas as situações; ele precisa escolher ações que maximizem o desempenho esperado dadas as informações disponíveis.

---

## 12. Medida de Desempenho (Performance Measure)

* **Termo original:** Performance Measure
* **Tradução/equivalente:** Medida de desempenho / critério de desempenho
* **Definição:** Critério utilizado para avaliar o quão bem um agente está atingindo seus objetivos.
* **Função no modelo:** Determina o que significa "bom comportamento" para o agente.
* **Relação com outros conceitos:** É fundamental para definir racionalidade. Um agente só pode ser considerado racional em relação a algum critério de desempenho.
* **Aspecto técnico:** A medida deve avaliar os resultados produzidos pelo agente no ambiente, e não simplesmente aquilo que o agente acredita estar fazendo.
* **Exemplo conceitual:** Para um táxi autônomo, a medida pode considerar segurança, tempo de viagem, conforto, legalidade e custo.

---

## 13. Conhecimento Prévio (Prior Knowledge)

* **Termo original:** Prior Knowledge
* **Tradução/equivalente:** Conhecimento prévio
* **Definição:** Informações que o agente possui antes de interagir com determinada situação do ambiente.
* **Função no modelo:** Permite que o agente tome decisões mesmo quando possui pouca experiência direta.
* **Relação com outros conceitos:** É um dos elementos considerados na racionalidade do agente.
* **Exemplo conceitual:** Um robô pode saber previamente que objetos físicos não podem ocupar o mesmo espaço e utilizar esse conhecimento para planejar seus movimentos.

---

## 14. Autonomia (Autonomy)

* **Termo original:** Autonomy
* **Tradução/equivalente:** Autonomia
* **Definição:** Grau em que o comportamento de um agente depende de sua própria experiência e percepção, em vez de depender exclusivamente do conhecimento fornecido pelo projetista.
* **Função no modelo:** Permite que o agente adapte seu comportamento com base na experiência adquirida.
* **Relação com outros conceitos:** Está fortemente relacionada à aprendizagem. Um agente que aprende pode compensar conhecimento inicial incompleto ou incorreto.
* **Exemplo conceitual:** Um robô que recebe regras básicas do fabricante, mas aprende com sua experiência quais caminhos são mais eficientes, possui maior autonomia.

---

# 15. Ambiente de Tarefa (Task Environment)

* **Termo original:** Task Environment
* **Tradução/equivalente:** Ambiente de tarefa
* **Definição:** Especificação do problema ou situação na qual um agente deve operar, incluindo aquilo que precisa ser alcançado e as condições do ambiente.
* **Função no modelo:** Define o contexto em que o agente será projetado e avaliado.
* **Relação com outros conceitos:** É descrito através do framework PEAS e também pode ser classificado segundo propriedades como observabilidade, determinismo, dinamismo e número de agentes.
* **Exemplo conceitual:** Para um táxi autônomo, o ambiente de tarefa envolve transportar passageiros com segurança e eficiência através de um ambiente rodoviário.

---

# 16. PEAS

* **Termo original:** PEAS
* **Tradução/equivalente:** Medida de desempenho, ambiente, atuadores e sensores
* **Definição:** Estrutura utilizada para especificar um ambiente de tarefa através de quatro componentes: **Performance measure, Environment, Actuators e Sensors**.
* **Função no modelo:** Permite definir claramente o problema que um agente deve resolver antes de projetar sua arquitetura.
* **Relação com outros conceitos:** Está diretamente relacionado ao Task Environment, ao agente racional e aos sensores e atuadores.
* **Exemplo conceitual — táxi autônomo:**

**P — Performance:** segurança, rapidez, conforto, legalidade e lucro.

**E — Environment:** ruas, veículos, pedestres, passageiros, semáforos, clima etc.

**A — Actuators:** direção, acelerador, freio, buzina etc.

**S — Sensors:** câmeras, GPS, radar, velocímetro etc.

---

# 17. Observabilidade

* **Termo original:** Observability
* **Tradução/equivalente:** Observabilidade
* **Definição:** Característica que determina quanto do estado relevante do ambiente pode ser percebido pelos sensores do agente.
* **Função no modelo:** Influencia a quantidade de informação que o agente possui para tomar decisões.
* **Relação com outros conceitos:** Determina se um ambiente é totalmente observável ou parcialmente observável.
* **Exemplo conceitual:** No xadrez, o estado do tabuleiro está completamente disponível para os jogadores. Já um carro autônomo não consegue observar tudo que acontece ao seu redor.

---

# 18. Ambiente Totalmente Observável (Fully Observable Environment)

* **Termo original:** Fully Observable
* **Tradução/equivalente:** Ambiente totalmente observável
* **Definição:** Ambiente no qual os sensores do agente fornecem todas as informações necessárias para determinar o estado relevante do ambiente.
* **Função no modelo:** Facilita a tomada de decisões porque o agente não precisa inferir informações ocultas.
* **Relação com outros conceitos:** É o oposto de um ambiente parcialmente observável.
* **Exemplo conceitual:** Um jogo de xadrez é considerado totalmente observável porque todas as peças e suas posições estão visíveis aos jogadores.

---

# 19. Ambiente Parcialmente Observável (Partially Observable Environment)

* **Termo original:** Partially Observable
* **Tradução/equivalente:** Ambiente parcialmente observável
* **Definição:** Ambiente no qual os sensores não fornecem todas as informações necessárias sobre o estado atual.
* **Função no modelo:** Exige que o agente utilize memória, inferência ou modelos internos para lidar com informações ausentes.
* **Relação com outros conceitos:** Está diretamente relacionado ao estado interno e aos agentes baseados em modelo.
* **Exemplo conceitual:** Um carro autônomo não consegue saber diretamente se existe um veículo atrás de uma parede ou de outro carro.

---

# 20. Determinismo (Deterministic Environment)

* **Termo original:** Deterministic
* **Tradução/equivalente:** Determinístico
* **Definição:** Um ambiente determinístico é aquele no qual o resultado de uma ação é completamente determinado pelo estado atual e pela própria ação.
* **Função no modelo:** Permite prever exatamente o resultado de uma ação quando o estado relevante é conhecido.
* **Relação com outros conceitos:** Ambientes não determinísticos ou estocásticos introduzem incerteza sobre os resultados das ações.
* **Exemplo conceitual:** Em um jogo de xadrez, mover uma peça de uma posição válida produz um resultado determinado pelas regras do jogo.

---

# 21. Ambiente Estocástico (Stochastic Environment)

* **Termo original:** Stochastic
* **Tradução/equivalente:** Estocástico / probabilístico
* **Definição:** Ambiente no qual o resultado de uma ação não é completamente previsível, mesmo quando o agente conhece o estado atual.
* **Função no modelo:** Obriga o agente a considerar diferentes resultados possíveis e suas probabilidades.
* **Relação com outros conceitos:** Está diretamente relacionado à incerteza e à Utility-Based Rationality.
* **Exemplo conceitual:** Um robô que tenta atravessar uma superfície escorregadia pode executar a mesma ação e obter resultados diferentes.

---

# 22. Ambiente Episódico (Episodic Environment)

* **Termo original:** Episodic
* **Tradução/equivalente:** Episódico
* **Definição:** Ambiente no qual a experiência do agente pode ser dividida em episódios independentes, de modo que a decisão tomada em um episódio não afeta significativamente os episódios seguintes.
* **Função no modelo:** Permite que o agente tome decisões considerando cada episódio de maneira relativamente independente.
* **Relação com outros conceitos:** É contrastado com ambientes sequenciais.
* **Exemplo conceitual:** Um sistema que classifica imagens individuais pode tratar cada imagem como um episódio separado.

---

# 23. Ambiente Sequencial (Sequential Environment)

* **Termo original:** Sequential
* **Tradução/equivalente:** Sequencial
* **Definição:** Ambiente no qual as ações tomadas anteriormente podem afetar as situações e decisões futuras.
* **Função no modelo:** Exige que o agente considere consequências de longo prazo.
* **Relação com outros conceitos:** Está relacionado a memória, planejamento, objetivos e agentes baseados em utilidade.
* **Exemplo conceitual:** No xadrez, uma jogada influencia todas as posições e possibilidades futuras da partida.

---

# 24. Ambiente Estático (Static Environment)

* **Termo original:** Static
* **Tradução/equivalente:** Estático
* **Definição:** Ambiente que não sofre alterações enquanto o agente está deliberando sobre sua próxima ação.
* **Função no modelo:** Facilita o processo de decisão porque o agente pode raciocinar sem que o ambiente mude durante esse processo.
* **Relação com outros conceitos:** É contrastado com ambientes dinâmicos.
* **Exemplo conceitual:** Um quebra-cabeça físico parado pode ser considerado estático enquanto o agente está decidindo o próximo movimento.

---

# 25. Ambiente Dinâmico (Dynamic Environment)

* **Termo original:** Dynamic
* **Tradução/equivalente:** Dinâmico
* **Definição:** Ambiente que pode mudar enquanto o agente está tomando uma decisão.
* **Função no modelo:** Exige que o agente considere que suas informações podem ficar desatualizadas rapidamente.
* **Relação com outros conceitos:** É particularmente importante em robótica e sistemas autônomos.
* **Exemplo conceitual:** Um carro autônomo opera em um ambiente dinâmico porque outros veículos e pedestres continuam se movimentando enquanto ele decide o que fazer.

---

# 26. Ambiente Discreto (Discrete Environment)

* **Termo original:** Discrete
* **Tradução/equivalente:** Discreto
* **Definição:** Ambiente no qual os estados, perceptos, ações ou eventos relevantes podem ser representados como elementos distintos e contáveis.
* **Função no modelo:** Simplifica a representação do ambiente e das ações disponíveis.
* **Relação com outros conceitos:** É contrastado com ambientes contínuos.
* **Exemplo conceitual:** O xadrez possui um conjunto discreto de posições e movimentos válidos.

---

# 27. Ambiente Contínuo (Continuous Environment)

* **Termo original:** Continuous
* **Tradução/equivalente:** Contínuo
* **Definição:** Ambiente no qual estados, ações ou percepções podem assumir valores dentro de intervalos contínuos.
* **Função no modelo:** Torna a representação e a tomada de decisão mais complexas.
* **Relação com outros conceitos:** É comum em sistemas físicos, robótica e veículos autônomos.
* **Exemplo conceitual:** A velocidade de um carro pode assumir inúmeros valores dentro de determinado intervalo, em vez de apenas algumas opções discretas.

---

# 28. Agente Único (Single-Agent)

* **Termo original:** Single-Agent Environment
* **Tradução/equivalente:** Ambiente de agente único
* **Definição:** Ambiente no qual apenas um agente está tomando decisões relevantes para a tarefa considerada.
* **Função no modelo:** Permite analisar o comportamento do agente sem considerar outros agentes como participantes ativos da tomada de decisão.
* **Exemplo conceitual:** Um sistema que resolve sozinho um quebra-cabeça pode ser tratado como um agente único.

---

# 29. Ambiente Multiagente (Multiagent Environment)

* **Termo original:** Multiagent Environment
* **Tradução/equivalente:** Ambiente multiagente
* **Definição:** Ambiente no qual existem múltiplos agentes cujas ações podem influenciar o desempenho uns dos outros.
* **Função no modelo:** Exige considerar o comportamento de outros agentes durante a tomada de decisão.
* **Relação com outros conceitos:** Pode envolver cooperação, competição ou uma combinação das duas.
* **Exemplo conceitual:** Uma partida de futebol envolve vários agentes, cada um tomando decisões que influenciam os demais.

---

# 30. Agente Reativo Simples (Simple Reflex Agent)

* **Termo original:** Simple Reflex Agent
* **Tradução/equivalente:** Agente reativo simples
* **Definição:** Agente que escolhe suas ações com base principalmente no percepto atual, utilizando regras do tipo condição–ação.
* **Função no modelo:** Implementa um comportamento simples e direto sem manter uma representação completa do histórico do ambiente.
* **Relação com outros conceitos:** É o tipo mais simples de agente apresentado no capítulo.
* **Aspecto técnico:** Pode ser representado por regras semelhantes a:

`SE condição → ENTÃO ação`

* **Exemplo conceitual:**

```text
SE sujeira → aspirar
SENÃO SE estiver em A → mover para B
SENÃO → mover para A
```

* **Limitação:** Pode apresentar comportamento inadequado em ambientes parcialmente observáveis, pois não possui memória suficiente para inferir informações ausentes.

---

# 31. Regra Condição–Ação (Condition-Action Rule)

* **Termo original:** Condition-Action Rule
* **Tradução/equivalente:** Regra condição–ação / regra se-então
* **Definição:** Regra que especifica uma ação a ser realizada quando determinada condição é satisfeita.
* **Função no modelo:** Constitui a base de decisão de agentes reativos simples.
* **Relação com outros conceitos:** É utilizada diretamente pelos Simple Reflex Agents.
* **Exemplo conceitual:**

`SE o semáforo estiver vermelho → frear`

---

# 32. Agente Reativo Baseado em Modelo (Model-Based Reflex Agent)

* **Termo original:** Model-Based Reflex Agent
* **Tradução/equivalente:** Agente reativo baseado em modelo
* **Definição:** Agente que mantém um estado interno do ambiente e utiliza um modelo de como o mundo funciona para interpretar seus perceptos.
* **Função no modelo:** Permite lidar melhor com ambientes parcialmente observáveis.
* **Relação com outros conceitos:** Utiliza estado interno, modelo de transição e modelo de sensores.
* **Aspecto técnico:** O agente combina o percepto atual com seu estado anterior para atualizar sua representação do mundo.
* **Exemplo conceitual:** Um carro pode manter uma estimativa de que existe outro veículo ao seu lado mesmo quando esse veículo momentaneamente sai do campo de visão.

---

# 33. Estado Interno (Internal State)

* **Termo original:** Internal State
* **Tradução/equivalente:** Estado interno
* **Definição:** Informação mantida pelo agente para representar aspectos relevantes do ambiente que não estão disponíveis diretamente no percepto atual.
* **Função no modelo:** Permite que o agente utilize memória e raciocine sobre informações observadas anteriormente.
* **Relação com outros conceitos:** É fundamental para Model-Based Reflex Agents.
* **Exemplo conceitual:** Um robô pode manter internamente a informação de que já visitou determinado corredor.

---

# 34. Modelo do Mundo (Model of the World)

* **Termo original:** Model of the World
* **Tradução/equivalente:** Modelo do mundo
* **Definição:** Conhecimento utilizado pelo agente para representar como o ambiente funciona e como ele muda ao longo do tempo.
* **Função no modelo:** Permite prever ou inferir estados que não podem ser observados diretamente.
* **Relação com outros conceitos:** É utilizado pelos agentes baseados em modelo e envolve informações sobre as consequências das ações e sobre a evolução independente do ambiente.
* **Exemplo conceitual:** Um carro pode possuir um modelo que relaciona a ação "girar o volante" com a mudança esperada na direção do veículo.

---

# 35. Modelo de Transição (Transition Model)

* **Termo original:** Transition Model
* **Tradução/equivalente:** Modelo de transição
* **Definição:** Conhecimento sobre como o estado do ambiente muda ao longo do tempo em função das ações do agente e da evolução do próprio ambiente.
* **Função no modelo:** Permite ao agente prever como determinada ação pode modificar o estado do mundo.
* **Relação com outros conceitos:** Faz parte do modelo utilizado pelos Model-Based Agents.
* **Exemplo conceitual:** Se um agente está em uma posição e executa "mover para a direita", o modelo de transição descreve qual será o estado esperado depois dessa ação.

---

# 36. Modelo de Sensor (Sensor Model)

* **Termo original:** Sensor Model
* **Tradução/equivalente:** Modelo de sensor
* **Definição:** Conhecimento sobre como o estado do ambiente é refletido nas percepções fornecidas pelos sensores.
* **Função no modelo:** Permite interpretar os perceptos e relacioná-los aos possíveis estados do ambiente.
* **Relação com outros conceitos:** Trabalha em conjunto com o modelo de transição em agentes baseados em modelo.
* **Exemplo conceitual:** Um sensor de câmera pode fornecer uma imagem que precisa ser interpretada para determinar se existe um veículo à frente.

---

# 37. Agente Baseado em Objetivos (Goal-Based Agent)

* **Termo original:** Goal-Based Agent
* **Tradução/equivalente:** Agente baseado em objetivos/metas
* **Definição:** Agente que utiliza informações sobre objetivos desejados para determinar quais ações devem ser tomadas.
* **Função no modelo:** Permite que o agente considere estados futuros e escolha ações que conduzam à realização de um objetivo.
* **Relação com outros conceitos:** Pode utilizar um modelo do mundo e está relacionado diretamente a busca e planejamento.
* **Exemplo conceitual:** Um GPS pode considerar como objetivo chegar a uma determinada cidade e procurar uma sequência de ações que leve até ela.
* **Aspecto técnico:** Diferentemente de um agente puramente reativo, o agente baseado em objetivos pode considerar consequências futuras.

---

# 38. Objetivo (Goal)

* **Termo original:** Goal
* **Tradução/equivalente:** Objetivo / meta
* **Definição:** Descrição de uma situação ou estado desejável que o agente pretende alcançar.
* **Função no modelo:** Fornece um critério para selecionar ações que levem o agente a uma situação desejada.
* **Relação com outros conceitos:** É o componente central dos Goal-Based Agents e está relacionado a busca e planejamento.
* **Exemplo conceitual:** Para um robô, um objetivo pode ser:

`"Chegar à sala B."`

O agente então escolhe ações que eventualmente permitam alcançar esse estado.

---

# 39. Busca (Search)

* **Termo original:** Search
* **Tradução/equivalente:** Busca
* **Definição:** Processo de explorar possíveis sequências de ações para encontrar uma que permita atingir um objetivo.
* **Função no modelo:** Permite que agentes baseados em objetivos encontrem caminhos para estados desejados.
* **Relação com outros conceitos:** Está diretamente relacionada aos Goal-Based Agents e ao planejamento.
* **Observação:** O livro desenvolve busca de forma mais aprofundada no Capítulo 3.
* **Exemplo conceitual:** Encontrar uma sequência de movimentos que leve um personagem de uma posição inicial até a saída de um labirinto.

---

# 40. Planejamento (Planning)

* **Termo original:** Planning
* **Tradução/equivalente:** Planejamento
* **Definição:** Processo de determinar uma sequência de ações que permita atingir um objetivo considerando o estado atual e as consequências das ações.
* **Função no modelo:** Permite que agentes orientados a objetivos pensem sobre ações futuras.
* **Relação com outros conceitos:** Está associado a Goal-Based Agents e ao modelo do mundo.
* **Exemplo conceitual:** Um robô pode planejar:

`pegar objeto → atravessar corredor → entrar na sala → entregar objeto`

---

# 41. Agente Baseado em Utilidade (Utility-Based Agent)

* **Termo original:** Utility-Based Agent
* **Tradução/equivalente:** Agente baseado em utilidade
* **Definição:** Agente que utiliza uma função de utilidade para avaliar quão desejáveis são diferentes estados ou resultados possíveis.
* **Função no modelo:** Permite escolher entre diferentes resultados possíveis, mesmo quando todos satisfazem um objetivo.
* **Relação com outros conceitos:** Está relacionado à Utility Function, Performance Measure, probabilidade e racionalidade.
* **Exemplo conceitual:** Um táxi pode ter como objetivo chegar ao destino, mas entre várias rotas possíveis pode preferir aquela que seja mais rápida, segura e barata.

---

# 42. Utilidade (Utility)

* **Termo original:** Utility
* **Tradução/equivalente:** Utilidade
* **Definição:** Valor numérico utilizado para representar a preferência do agente por determinado estado ou resultado.
* **Função no modelo:** Permite comparar resultados diferentes e determinar qual é mais desejável.
* **Relação com outros conceitos:** É utilizada por agentes baseados em utilidade e está relacionada à medida de desempenho.
* **Exemplo conceitual:**

```text
Estado A → utilidade 80
Estado B → utilidade 50
Estado C → utilidade 20
```

O agente tende a preferir A, assumindo que valores maiores representam resultados melhores.

---

# 43. Função de Utilidade (Utility Function)

* **Termo original:** Utility Function
* **Tradução/equivalente:** Função de utilidade
* **Definição:** Função que associa estados ou sequências de estados a valores que representam a preferência do agente.
* **Função no modelo:** Permite que o agente determine quantitativamente quais resultados são mais desejáveis.
* **Relação com outros conceitos:** É uma espécie de representação interna da medida de desempenho e é fundamental para Utility-Based Agents.
* **Aspecto técnico:** Em ambientes incertos, o agente pode escolher a ação que maximize a **utilidade esperada**, considerando diferentes resultados possíveis e suas probabilidades.
* **Exemplo conceitual:**

`U(estado) → valor`

Uma rota pode possuir utilidade maior por ser mais rápida e segura.

---

# 44. Utilidade Esperada (Expected Utility)

* **Termo original:** Expected Utility
* **Tradução/equivalente:** Utilidade esperada
* **Definição:** Valor médio esperado de uma ação considerando os possíveis resultados e suas respectivas probabilidades.
* **Função no modelo:** Permite que um agente racional tome decisões em situações de incerteza.
* **Relação com outros conceitos:** Está diretamente relacionada a agentes baseados em utilidade e ambientes estocásticos.
* **Exemplo conceitual:**

Se uma ação possui:

`50% → utilidade 100`

`50% → utilidade 20`

então:

`EU = 0,5 × 100 + 0,5 × 20 = 60`

A ação possui utilidade esperada igual a `60`.

---

# 45. Agente de Aprendizagem (Learning Agent)

* **Termo original:** Learning Agent
* **Tradução/equivalente:** Agente de aprendizagem
* **Definição:** Agente capaz de melhorar seu comportamento através da experiência.
* **Função no modelo:** Permite que o agente modifique seus componentes internos para melhorar seu desempenho ao longo do tempo.
* **Relação com outros conceitos:** Está relacionado à autonomia, feedback, desempenho e experiência.
* **Exemplo conceitual:** Um agente que inicialmente escolhe rotas ruins pode aprender, após várias experiências, quais caminhos normalmente são mais eficientes.

---

# 46. Elementos do Agente de Aprendizagem

O livro descreve um agente de aprendizagem através de quatro componentes conceituais principais:

### Learning Element

* **Tradução:** Elemento de aprendizagem.
* **Função:** Responsável por realizar mudanças no agente para melhorar seu desempenho.

### Performance Element

* **Tradução:** Elemento de desempenho.
* **Função:** Seleciona as ações que serão executadas.

### Critic

* **Tradução:** Crítico / avaliador.
* **Função:** Avalia o comportamento produzido pelo agente e fornece feedback ao elemento de aprendizagem.

### Problem Generator

* **Tradução:** Gerador de problemas.

* **Função:** Sugere ações exploratórias que podem gerar novas experiências e ajudar o agente a descobrir estratégias melhores.

* **Relação entre os componentes:**

```text
Ambiente
   ↓
Percepções
   ↓
Performance Element
   ↓
Ação
   ↓
Ambiente

        ↓ feedback

      Critic
        ↓
Learning Element
        ↓
melhora o agente

Problem Generator
        ↓
sugere exploração
```

---

# 47. Feedback

* **Termo original:** Feedback
* **Tradução/equivalente:** Retroalimentação / feedback
* **Definição:** Informação utilizada pelo agente para avaliar os resultados de seu comportamento e modificar suas estratégias.
* **Função no modelo:** É fundamental para agentes capazes de aprender com a experiência.
* **Relação com outros conceitos:** Está relacionado ao Critic, Learning Element e Performance Measure.
* **Exemplo conceitual:** Se um agente tenta uma rota e demora muito para chegar ao destino, essa experiência pode ser utilizada para modificar suas decisões futuras.

---

# 48. Modelo-Baseado (Model-Based)

* **Termo original:** Model-Based
* **Tradução/equivalente:** Baseado em modelo
* **Definição:** Característica de um agente que utiliza uma representação interna de como o ambiente funciona.
* **Função no modelo:** Permite ao agente inferir aspectos do ambiente que não estão disponíveis diretamente através dos sensores.
* **Relação com outros conceitos:** Está diretamente relacionado ao Internal State, Transition Model e Sensor Model.
* **Exemplo conceitual:** Um robô pode utilizar um mapa interno para saber que uma determinada área existe mesmo quando ela não está atualmente visível.

---

# 49. Reflexo (Reflex)

* **Termo original:** Reflex
* **Tradução/equivalente:** Reflexo / comportamento reativo
* **Definição:** Estratégia na qual uma ação é escolhida diretamente em resposta às condições percebidas.
* **Função no modelo:** Permite implementar comportamentos rápidos e simples.
* **Relação com outros conceitos:** Constitui a base dos Simple Reflex Agents.
* **Exemplo conceitual:**

`SE detectar obstáculo → virar`

Esse comportamento não exige necessariamente um planejamento completo do futuro.

---

# 50. Exploração (Exploration)

* **Termo original:** Exploration
* **Tradução/equivalente:** Exploração
* **Definição:** Comportamento de realizar ações que podem fornecer novas informações ou experiências ao agente.
* **Função no modelo:** É especialmente importante para agentes de aprendizagem, pois permite adquirir conhecimento sobre o ambiente.
* **Relação com outros conceitos:** Está relacionada ao Problem Generator e à autonomia.
* **Exemplo conceitual:** Um robô pode escolher explorar um corredor desconhecido para descobrir se existe uma rota mais eficiente.

---

## Resumo das relações entre os principais conceitos

O Capítulo 2 pode ser entendido como uma sequência lógica:

```text
                    AMBIENTE
                       ↑
                       │
                    Ações
                       │
                   Atuadores
                       ↑
                    AGENTE
                       ↓
                    Sensores
                       ↓
                   Perceptos
                       ↓
              Programa de Agente
                       ↓
                     Ação
```

A partir daí, o capítulo introduz a ideia de **racionalidade**:

```text
Perceptos + Conhecimento + Ações disponíveis
                    ↓
             Agente Racional
                    ↓
       maximizar desempenho esperado
```

E mostra como diferentes arquiteturas de agentes aumentam progressivamente sua capacidade:

```text
Agente Reativo Simples
          ↓
Agente Reativo Baseado em Modelo
          ↓
Agente Baseado em Objetivos
          ↓
Agente Baseado em Utilidade
          ↓
Agente de Aprendizagem
```




