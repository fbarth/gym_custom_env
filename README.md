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

## Uso do ambiente GridWorld para problemas de Coverage Path Planning

**Sugestão**: considerando a última versão do ambiente GridWorld, com renderização e obstáculos, altere a função de *reward* e o que mais for necessário para que o agente aprenda a fazer *Coverage Path Planning* (CPP) em um ambiente 2D com obstáculos.

---

## Uso do ambiente GridWorld para Coverage Path Planning (CPP)

> Esta seção substitui a proposta original da última seção do README.

### Função de Reward Original (`grid_world_obstacles.py`)

O ambiente original recompensa o agente por **chegar a um alvo fixo** o mais rápido possível:

| Evento | Reward |
|---|---|
| Agente alcança o alvo | `+10.0` |
| Cada passo executado | `(dist_anterior - dist_atual) - 0.1` |
| Timeout (max_steps atingido) | `-10.0` |

A componente `(dist_anterior - dist_atual)` é um *shaping reward* baseado na distância Manhattan/Euclidiana até o alvo, o que incentiva o agente a se aproximar do objetivo. Esta função é adequada para tarefas de **navegação ponto a ponto**, mas não incentiva exploração ou cobertura do grid.

---

### Nova Função de Reward para CPP (`grid_world_cpp.py`)

Inspirada nos trabalhos:
- Theile et al. (2020) — *A Deep Reinforcement Learning Approach for the Patrolling Problem of Water Resources Through ASVs*
- Survey on Coverage Path Planning for Mobile Robots in Dynamic Environments (2025)

O agente deve **cobrir todas as células acessíveis** do grid (sem obstáculos). Não existe mais um alvo fixo. Em vez disso, o estado inclui um mapa binário de células visitadas, e o episódio termina com sucesso quando 100% das células acessíveis forem visitadas.

| Evento | Reward |
|---|---|
| Move para uma célula **nova** (não visitada) | `+1.0` |
| Move para uma célula **já visitada** (revisita) | `-0.5` |
| Penalidade de passo (a cada step) | `-0.05` |
| Bônus de **cobertura total** (100% das células) | `+10.0` |
| Timeout sem cobertura total | `-5.0` |

A recompensa positiva por novas células incentiva exploração; a penalidade por revisita desincentiva caminhos redundantes; o bônus de cobertura total reforça o objetivo final.

#### Como executar

```bash
# 1. Criar e ativar o ambiente virtual (primeira vez)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Testar o ambiente com agente aleatório (com renderização)
python run_grid_world_cpp.py

# 3. Testar sem renderização (mais rápido, só métricas no terminal)
python run_grid_world_cpp.py headless

# 4. Treinar o agente PPO
python train_grid_world_cpp.py train

# 5. Avaliar o agente treinado (100 episódios, métricas no terminal)
python train_grid_world_cpp.py test

# 6. Visualizar o agente treinado (1 episódio com renderização)
python train_grid_world_cpp.py run

# 7. Acompanhar o treinamento via TensorBoard
tensorboard --logdir log/grid_world_cpp
```
