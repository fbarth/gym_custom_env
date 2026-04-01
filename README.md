# Criando ambientes customizados usando a biblioteca Gymnasium

O objetivo deste repositório é fornecer alguns exemplos de ambientes customizados criados 
usando a biblioteca Gymnasium. 

Você pode usar este arquivo README.md como um handout para entender como implementar ambientes customizados e como utilizá-los.

## Instalação

Para começar a usar este repositório você precisa clonar o repositório e instalar as dependências necessárias. Você pode fazer isso usando os seguintes comandos depois de clonar o repositório:

```bash
python -m venv venv # para criar um ambiente virtual
source venv/bin/activate # para ativar o ambiente virtual
pip install -r requirements.txt # para instalar as dependências
```

## Primeiro exemplo: ambiente GridWorld sem renderização

O primeiro exemplo é um ambiente simples de grid world. O agente pode se mover para cima, baixo, esquerda ou direita. O objetivo do agente é chegar ao objetivo (goal) o mais rápido possível. O ambiente é definido na classe `GridWorldEnv` que está no arquivo `grid_world.py` dentro da pasta `gymnasium_env`. 

O código deste arquivo é baseado no tutorial disponível em [https://gymnasium.farama.org/introduction/create_custom_env/](https://gymnasium.farama.org/introduction/create_custom_env/). Este código tem todos os métodos necessários para criar um ambiente: `__init__`, `reset` e `step`. Só não tem o médoto `render` que é responsável por mostrar visualmente o ambiente.  

Os arquivos listados abaixo utilizam o ambiente `GridWorldEnv`: 

* `run_grid_world_v0.py`: registra o ambiente e executa um episódio, onde o comportamento do agente é aleatório.
* `run_grid_world_v0_wrapper.py`: utiliza a mesma base de código do arquivo anterior, além disso, faz uso de um wrapper para modificar a forma como o estado é retornado pelo ambiente e tratado pelo agente. 

**Questão**: Qual é a diferença entre o estado retornado pelo ambiente e o estado retornado pelo ambiente com o uso do wrapper? O que cada variável representa?

* `train_grid_world_v0.py`: faz uso do algoritmo PPO da biblioteca Stable Baselines3 para treinar um agente para atuar no ambiente `GridWorldEnv`. 

**Proposta**: 

* Execute o comando:

```bash
python train_grid_world_render_v0.py train
```

* Visualize a curva de aprendizado usando o plugin do tensorboard com os dados armazenados na pasta `log`. 

* Execute diversas vezes o comando: 

```bash
python train_grid_world_render_v0.py test
```

para visualizar se o agente aprendeu a melhor política. 


## Segundo exemplo: ambiente GridWorld com renderização

O segundo exemplo é o mesmo ambiente de grid world, mas agora a implementação do ambiente tem o método `render` que mostra visualmente o ambiente. A implementação deste ambiente está no arquivo `grid_world_render.py` dentro da pasta `gymnasium_env`.

Os arquivos que utilizam o ambiente `GridWorldEnv` com renderização são:

* `run_grid_world_render_v0.py`: registra o ambiente e executa um episódio, onde o comportamento do agente é aleatório.
* `run_grid_world_render_v0_wrapper.py`: utiliza a mesma base de código do arquivo anterior, além disso, faz uso de um wrapper para modificar a forma como o estado é retornado pelo ambiente e tratado pelo agente.
* `train_grid_world_render_v0.py`: faz uso do algoritmo PPO da biblioteca Stable Baselines3 para treinar um agente para atuar no ambiente `GridWorldEnv` com renderização.

Este último arquivo tem um código mais completo, pois o agente é treinado para atuar em um ambiente que tem uma representação visual, o modelo treinado é salvo e depois carregado para fazer uma execução do ambiente. Os dados sobre o treinamento do agente são salvos para depois serem utilizados pelo `tensorboard`.

## Terceiro exemplo: ambiente GridWorld em 3D

O terceiro exemplo é uma extensão do ambiente de grid world para um ambiente 3D. O agente pode se mover para cima, baixo, esquerda, direita, frente e trás. O objetivo do agente é chegar ao objetivo (goal) o mais rápido possível. O ambiente é definido na classe `GridWorldEnv` que está no arquivo `grid_world_3D.py` dentro da pasta `gymnasium_env`.

O arquivo que utiliza o ambiente `GridWorldEnv` em 3D é:
* `train_grid_world_3D.py`: faz uso do algoritmo PPO da biblioteca Stable Baselines3 para treinar um agente para atuar no ambiente `GridWorldEnv` em 3D.

Existem 3 (três) formas de uso do script `train_grid_world_3D.py`:
* `python train_grid_world_3D.py train`: treina o agente e salva o modelo treinado na pasta `data` e os logs na pasta `log`.    
* `python train_grid_world_3D.py test`: carrega o modelo treinado e executa 100 episódios, calculando o percentual de sucesso do agente, entre outras métricas.
* `python train_grid_world_3D.py run`: carrega o modelo treinado e executa um único episódio, mostrando a renderização do ambiente 3D.

Para que a renderização deste ambiente aconteça, é necessário ter a biblioteca `tkinter` instalada. No Ubuntu, você pode instalar esta biblioteca com o comando:

```bash
sudo apt-get install python3-tk
```

**Importante**: esta renderização 3D foi testada apenas no sistema operacional Ubuntu.


## Quarto exemplo: ambiente GridWorld com obstáculos

O quarto exemplo é uma extensão do ambiente de grid world para incluir obstáculos. O agente deve navegar pelo ambiente evitando os obstáculos para alcançar o objetivo. O ambiente é definido na classe `GridWorldEnv` que está no arquivo `grid_world_obstacles.py` dentro da pasta `gymnasium_env`.

Para executar o treinamento do agente no ambiente com obstáculos, execute o comando:

```bash
python train_grid_world_obstacles.py train
```

Para testar o agente treinado no ambiente com obstáculos, execute o comando:

```bash
python train_grid_world_obstacles.py test
```

Esta funcionalidade irá executar o agente treinado em 100 episódios e calcular o percentual de sucesso do agente, entre outras métricas. 

Também é possível executar o agente treinado em um único episódio, para isso execute o comando:

```bash
python train_grid_world_obstacles.py run
```

## Quinto exemplo: Coverage Path Planning (CPP)

Coverage Path Planning (CPP) é um problema clássico de planejamento onde o objetivo é encontrar um caminho que cubra todas as células (ou pontos) de um ambiente. Aplicações incluem robótica de serviço (aspiração autônoma), mapeamento de terrenos, inspeção de infraestrutura e agricultura de precisão.

### Função de reward original (navegação com obstáculos)

O ambiente `grid_world_obstacles.py` foi projetado para que o agente **alcance um alvo fixo**. A função de reward é:

| Evento | Reward |
|--------|--------|
| Alcançar o objetivo (target) | `+10.0` |
| Cada passo (shaping de distância) | `prev_dist − cur_dist − 0.1` |
| Esgotar `max_steps` sem alcançar o objetivo | `−10.0` |

O shaping de distância (`prev_dist − cur_dist`) encoraja o agente a se aproximar do alvo a cada passo, ao mesmo tempo que o custo de `-0.1` por passo desincentiva rotas longas.

### Nova função de reward (CPP)

Para o CPP, **não existe alvo fixo**: o agente deve visitar todas as células livres do grid. A função de reward foi redesenhada com base nos trabalhos:

- *A Deep Reinforcement Learning Approach for the Patrolling Problem of Water Resources Through Autonomous Surface Vehicles: The Ypacarai Lake Case* — que utiliza recompensas baseadas na novidade das células visitadas para incentivar cobertura distribuída.
- *A Comprehensive Survey on Coverage Path Planning for Mobile Robots in Dynamic Environments* — que sistematiza estratégias de reward shaping para CPP, incluindo penalidades por revisita e bônus de conclusão.

| Evento | Reward |
|--------|--------|
| Visitar uma célula **nova** (não visitada) | `+1.0` |
| Revisitar uma célula já visitada | `−0.5` |
| Bônus por **cobertura total** de todas as células livres | `+50.0` |
| Custo por passo (incentiva eficiência) | `−0.05` |
| Esgotar `max_steps` sem cobertura total | `−10.0` |

A combinação de recompensa positiva por célula nova e penalidade por revisita direciona o agente para explorar sistematicamente o grid em vez de circular pelas mesmas células. O bônus de `+50.0` ao concluir a cobertura é suficientemente alto para dominar a soma das recompensas individuais, garantindo que o agente aprenda a priorizar a conclusão da tarefa.

### Observação estendida

O espaço de observação foi ampliado para fornecer ao agente informação sobre quais células já foram visitadas:

```
[agent_x, agent_y, visited_map[0..size×size−1], neighbors[0..3]]
```

O `visited_map` codifica: `1` = célula visitada, `0` = célula livre não visitada, `−1` = obstáculo.

### Como executar

Para visualizar o comportamento de um **agente aleatório** em um grid 5×5:

```bash
python run_grid_world_cpp.py
```

Para **treinar** um agente PPO no ambiente CPP:

```bash
python train_grid_world_cpp.py train
```

Para **testar** o agente treinado em 100 episódios (reporta taxa de cobertura completa e cobertura média):

```bash
python train_grid_world_cpp.py test
```

Para **visualizar** um único episódio com o agente treinado:

```bash
python train_grid_world_cpp.py run
```
