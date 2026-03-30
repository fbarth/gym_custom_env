# Criando ambientes customizados usando a biblioteca Gymnasium

## Aluno: Matheus Castellucci

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

O *Coverage Path Planning* (CPP) é um problema clássico de planejamento onde o objetivo do agente é cobrir todas as células livres de um ambiente, evitando obstáculos e revisitas desnecessárias. Este problema tem aplicações em robótica (limpeza autônoma, mapeamento), logística e agricultura de precisão.

Para adaptar o ambiente GridWorld ao CPP, é necessário alterar a função de *reward* e o espaço de observação para que o agente aprenda a cobrir todo o espaço livre do grid em vez de navegar até um alvo fixo.

### Função de reward original (Goal-reaching)

A função de *reward* original foi projetada para o problema de navegação até um alvo (*goal-reaching*). Ela está implementada no arquivo `grid_world_obstacles.py` e funciona da seguinte forma:

| Situação | Reward |
|---|---|
| Agente alcança o alvo | `+10.0` |
| Cada passo sem alcançar o alvo | `distância_anterior - distância_atual - 0.1` |
| Número máximo de passos atingido sem alcançar o alvo | `-10.0` |

A componente `distância_anterior - distância_atual` é positiva quando o agente se aproxima do alvo e negativa quando se afasta, incentivando o agente a reduzir a distância euclidiana ao objetivo. O termo `-0.1` penaliza cada passo para incentivar eficiência.

Esta função **não é adequada para CPP** porque pressupõe um alvo fixo e penaliza exploração — comportamento oposto ao desejado no CPP.

---

### Versão 1 (CPP v1) — função de reward baseada em novidade de cobertura

A primeira versão do ambiente CPP está implementada no arquivo `grid_world_cpp.py`. A função de *reward* foi redesenhada para incentivar a cobertura máxima com o mínimo de revisitas, sem alvo fixo.

**Função de reward:**

| Situação | Reward |
|---|---|
| Agente visita uma célula nova (ainda não coberta) | `+1.0` |
| Agente revisita uma célula já visitada | `-0.5` |
| Penalidade por passo (incentiva eficiência) | `-0.1` |
| Agente cobre 100% das células livres (episódio encerrado) | `+10.0` (bônus) |
| Número máximo de passos atingido sem cobertura total | `-5.0` |

**Alterações no ambiente em relação ao goal-reaching:**

- **Estado (*observation*)**: inclui a posição do agente (2 valores), os 4 vizinhos bloqueados (4 valores) e um mapa binário flattened das células já visitadas (size² valores). A posição do alvo foi removida pois não existe no CPP.
- **Condição de término**: o episódio termina quando todas as células livres forem visitadas, ou quando o número máximo de passos for atingido.
- **Inicialização**: no `reset`, o mapa de visitados é zerado e a célula inicial do agente já é marcada como visitada.

**Resultados com PPO (grid 5×5, 3 obstáculos, 1.5M timesteps):**

| Métrica | Resultado |
|---|---|
| Taxa de sucesso (100% cobertura) | 24% |
| Cobertura média | 79.32% |

**Limitação identificada:** o agente aprendia a cobrir a maior parte do grid rapidamente (chegando frequentemente a 95%), mas travava em *loop* ao buscar a(s) última(s) célula(s) não visitadas. Isso ocorre porque a observação não fornecia informação sobre *onde* estavam as células restantes — o agente sabia *que* havia células por cobrir, mas não *onde*.

Para executar o ambiente v1 com agente aleatório:

```bash
python run_grid_world_cpp.py
```

---

### Versão 2 (CPP v2) — adição da célula não visitada mais próxima à observação

A segunda versão, implementada em `grid_world_cpp_v2.py`, mantém a mesma função de *reward* da v1 e resolve o problema da última célula adicionando ao espaço de observação a **posição da célula livre não visitada mais próxima** (distância de Manhattan).

Esta ideia é diretamente inspirada no trabalho de Schmid et al. (2025), *"A Deep Reinforcement Learning Approach for the Patrolling Problem of Water Resources Through Autonomous Surface Vehicles: The Ypacarai Lake Case"* (IEEE Access, doi:10.1109/ACCESS.2025.10946186), que utiliza informações de proximidade e novidade de cobertura para guiar o aprendizado do agente. A abordagem é complementada pelos princípios gerais levantados em *A Comprehensive Survey on Coverage Path Planning for Mobile Robots in Dynamic Environments*.

**O que mudou na observação (v1 → v2):**

| Componente | v1 | v2 |
|---|---|---|
| Posição do agente | ✓ (2 valores) | ✓ (2 valores) |
| Vizinhos bloqueados | ✓ (4 valores) | ✓ (4 valores) |
| Célula não visitada mais próxima | ✗ | ✓ (2 valores) |
| Mapa de células visitadas | ✓ (size² valores) | ✓ (size² valores) |

Com essa informação, o agente passa a ter um **sub-objetivo dinâmico**: em vez de vagar aleatoriamente em busca das últimas células, ele recebe a direção para onde deve ir a cada passo.

A função de *reward* permanece idêntica à v1.

**Resultados com PPO (grid 5×5, 3 obstáculos, 3M timesteps):**

| Métrica | v1 | v2 |
|---|---|---|
| Taxa de sucesso (100% cobertura) | 24% | **89%** |
| Cobertura média | 79.32% | **96.82%** |

**Visualização:** a célula não visitada mais próxima é destacada em amarelo na renderização, tornando o comportamento do agente mais fácil de interpretar.

Para executar o ambiente v2 com agente aleatório:

```bash
python run_grid_world_cpp_v2.py
```

Para treinar e testar o agente no ambiente v2:

```bash
python train_grid_world_cpp_v2.py train
python train_grid_world_cpp_v2.py test
python train_grid_world_cpp_v2.py run
```
