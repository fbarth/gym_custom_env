# 🤖 Ambientes Customizados com Gymnasium

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-latest-green.svg)](https://gymnasium.farama.org/)

Repositório com exemplos práticos de **ambientes customizados** criados com a biblioteca [Gymnasium](https://gymnasium.farama.org/). Ideal para aprender RL e prototipar novos ambientes.

**👥 Autoras**: Julia Almeida, Laura Pontirolli, Esther Cunha

## 📋 Conteúdo

1. [Instalação](#instalação)
2. [Ambientes Disponíveis](#ambientes-disponíveis)
3. [Guia de Uso Rápido](#guia-de-uso-rápido)
4. [Estrutura do Projeto](#estrutura-do-projeto)
5. [Função de Reward](#função-de-reward)
6. [Troubleshooting](#troubleshooting)

---

## 🚀 Instalação

### Pré-requisitos
- Python 3.8 ou superior
- pip ou conda

### Setup do Ambiente

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Nota**: Se receber erro de permissão no Windows, execute como administrador ou altere a política de execução:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## 📦 Ambientes Disponíveis

### 1️⃣ GridWorld Básico (sem renderização)
**Arquivo**: `gymnasium_env/grid_world.py`

O ambiente mais simples: um grid 2D onde o agente se move em 4 direções para atingir um alvo.

| Propriedade | Valor |
|------------|-------|
| Dimensão | 2D (grid NxN) |
| Ações | 4 (cima, baixo, esquerda, direita) |
| Recompensa | +1 ao atingir alvo, 0 caso contrário |
| Observação | Localização do agente e alvo |

**Scripts de execução:**
- `run_grid_world_v0.py` - Episódio com ações aleatórias
- `run_grid_world_v0_wrapper.py` - Usando wrappers para transformação de estado
- `train_grid_world_v0.py` - Treinar com PPO (Stable Baselines3)

**Questão desafiadora**: Qual é a diferença entre a observação retornada diretamente e após aplicar um wrapper?

---

### 2️⃣ GridWorld com Renderização
**Arquivo**: `gymnasium_env/grid_world_render.py`

Versão visual do GridWorld básico com renderização pygame.

| Propriedade | Valor |
|------------|-------|
| Renderização | Pygame (visual) |
| Dimensão | 2D |
| Ações | 4 |
| Taxa FPS | 4 fps |

**Scripts:**
- `run_grid_world_render_v0.py` - Visualização com ações aleatórias
- `train_grid_world_render_v0.py` - Treinar e visualizar com TensorBoard

**Teste o modelo treinado:**
```bash
python train_grid_world_render_v0.py train    # Treina
python train_grid_world_render_v0.py test     # Avalia
tensorboard --logdir ./log                    # Visualiza métricas
```

---

### 3️⃣ GridWorld em 3D
**Arquivo**: `gymnasium_env/grid_world_3D.py`

Extensão do GridWorld para 3 dimensões. O agente se move em 6 direções (cima, baixo, esquerda, direita, frente, trás).

| Propriedade | Valor |
|------------|-------|
| Dimensão | 3D (grid NxNxN) |
| Ações | 6 |
| Recompensa | +1.0 ao atingir, -1.0 se exceder max_steps |
| Renderização | OpenGL/Tkinter |

**Modos de uso:**
```bash
python train_grid_world_3D.py train     # Treina o modelo
python train_grid_world_3D.py test      # Testa em 100 episódios
python train_grid_world_3D.py run       # Executa 1 episódio com renderização
```

**Dependência adicional (Linux/Ubuntu):**
```bash
sudo apt-get install python3-tk
```

⚠️ **Nota**: A renderização 3D foi testada apenas em Ubuntu.

---

### 4️⃣ GridWorld com Obstáculos
**Arquivo**: `gymnasium_env/grid_world_obstacles.py`

GridWorld realista com obstáculos. O agente deve desviar de obstáculos para atingir o alvo.

| Propriedade | Valor |
|------------|-------|
| Dimensão | 2D com obstáculos |
| Ações | 4 |
| Colisão | Agente permanece no lugar |
| Reward Shaping | Baseado em distância |

**Função de Reward:**
```
• Atingir alvo:           +10.0
• Aproximar do alvo:      prev_dist - curr_dist - 0.1
• Colisão com obstáculo:  (agente fica parado)
• Esgotar max_steps:      -10.0
```

**Execução:**
```bash
python train_grid_world_obstacles.py train     # Treina
python train_grid_world_obstacles.py test      # Avalia (100 episódios)
python train_grid_world_obstacles.py run       # Visualiza 1 episódio
```

---

### 5️⃣ Coverage Path Planning (CPP) 
**Arquivo**: `gymnasium_env/grid_world_cpp.py`

Problema avançado: cobrir todas as células livres do grid (aplicações: limpeza autônoma, inspeção, mapeamento).

| Propriedade | Valor |
|------------|-------|
| Objetivo | Visitar TODAS as células livres |
| Recompensa | Baseada em novidade + cobertura |
| Observação | Mapa de células visitadas |

**Função de Reward Detalhada:**

| Evento | Reward | Justificativa |
|--------|--------|---------------|
| Visitar célula **nova** | +1.0 | Incentiva exploração |
| Revisitar (deliberadamente) | -0.5 | Desincentiva loops |
| Colidir com obstáculo | -0.3 | Diferencia tentativa bloqueada |
| **Cobertura total** (100%) | +50.0 | Bônus por conclusão |
| Por passo | -0.05 | Incentiva eficiência |
| Exceder max_steps | -10.0 | Penalidade por timeout |

**Inspiration:** Baseado em trabalhos de RL para robótica autônoma e planejamento de trajetórias.

---

## ⚡ Guia de Uso Rápido

### Exemplo 1: Treinar um agente simples
```bash
python train_grid_world_v0.py
```

### Exemplo 2: Treinar e visualizar
```bash
python train_grid_world_render_v0.py train
python train_grid_world_render_v0.py run
```

### Exemplo 3: Coverage Path Planning
```bash
python train_grid_world_cpp.py train
python train_grid_world_cpp.py test
```

---

## 📁 Estrutura do Projeto

```
gym_custom_env/
├── gymnasium_env/              # Environments
│   ├── grid_world.py          # 2D básico
│   ├── grid_world_render.py   # 2D com Pygame
│   ├── grid_world_3D.py       # 3D
│   ├── grid_world_obstacles.py # 2D com obstáculos
│   └── grid_world_cpp.py      # Coverage Path Planning
├── train_*.py                 # Scripts de treinamento
├── run_*.py                   # Scripts de teste/visualização
├── results_*.ipynb            # Notebooks de análise
├── requirements.txt           # Dependências
└── README.md                  # Este arquivo
```

---

## 🎯 Função de Reward

### GridWorld Básico
Reward simples: `+1` ao atingir, `0` caso contrário.

### GridWorld com Obstáculos
Usa reward shaping de distância:
```
• Atingir alvo:           +10.0
• Aproximar do alvo:      prev_dist - curr_dist - 0.1
• Esgotar max_steps:      -10.0
```

### Coverage Path Planning (CPP)

O `visited_map` codifica: `1` = célula visitada, `0` = célula livre não visitada, `−1` = obstáculo.

| Evento | Reward | Justificativa |
|--------|--------|---------------|
| Visitar célula **nova** | +1.0 | Incentiva exploração |
| Revisitar (deliberadamente) | -0.5 | Desincentiva loops |
| Colidir com obstáculo | -0.3 | Diferencia tentativa bloqueada |
| **Cobertura total** (100%) | +50.0 | Bônus por conclusão |
| Por passo | -0.05 | Incentiva eficiência |
| Exceder max_steps | -10.0 | Penalidade por timeout |

**Intuição:** A combinação de recompensa positiva por novidade + penalidade por revisita direciona exploração sistemática. A penalidade por colisão é mais branda que por revisita deliberada, permitindo diferenciação.

---

## ⚙️ Como Executar

### GPP (Coverage Path Planning)

Agente aleatório em grid 5×5:
```bash
python run_grid_world_cpp.py
```

Treinar um agente PPO:
```bash
python train_grid_world_cpp.py train
```

Testar em 100 episódios (taxa de cobertura completa + média):
```bash
python train_grid_world_cpp.py test
```

Visualizar um único episódio:
```bash
python train_grid_world_cpp.py run
```

---

## 🛠️ Troubleshooting

### ❌ Erro: "ModuleNotFoundError: No module named 'gymnasium'"
**Solução:**
```bash
pip install gymnasium stable-baselines3 pygame
```

### ❌ Erro: "No module named 'tkinter'" (ao rodar grid_world_3D)
**Linux/Ubuntu:**
```bash
sudo apt-get install python3-tk
```

**macOS:**
```bash
brew install python-tk
```

**Windows:** Geralmente já vem instalado. Se não, reinstale Python e marque "tcl/tk" na instalação.

### ❌ Renderização não funciona
- Verifique se Pygame está instalado: `pip install pygame`
- Para 3D no WSL2/Windows, configure um servidor X como VcXsrv
- GUI pode não funcionar em ambientes headless (servidores sem display)

### ❌ Treino muito lento
- Reduza o tamanho do grid: `size=5` ao invés de `size=10`
- Reduza o número de obstáculos: `obs_quantity=2` ao invés de `obs_quantity=5`
- Reduza `max_steps` para episódios mais curtos

### ❌ PowerShell: Erro de permissão
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📚 Tecnologias Utilizadas

| Biblioteca | Versão | Propósito |
|-----------|--------|----------|
| Gymnasium | ≥0.26 | Framework para ambientes RL |
| Stable Baselines3 | ≥2.0 | Algoritmos RL (PPO) |
| PyGame | ≥2.0 | Renderização 2D |
| NumPy | ≥1.21 | Computação numérica |
| TensorBoard | ≥2.13 | Visualização de logs |

---

## 🎓 Referências e Inspiração

### Papers sobre CPP
- *A Deep Reinforcement Learning Approach for the Patrolling Problem of Water Resources Through Autonomous Surface Vehicles: The Ypacarai Lake Case*
- *A Comprehensive Survey on Coverage Path Planning for Mobile Robots in Dynamic Environments*

### Recursos Externos
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable Baselines3 Guide](https://stable-baselines3.readthedocs.io/)
- [Tutorial Oficial de Custom Environments](https://gymnasium.farama.org/introduction/create_custom_env/)

---

## 📝 Notas Importantes

- ✅ Todos os ambientes seguem a API Gymnasium (v0.26+)
- ✅ Compatível com Stable Baselines3 e qualquer algoritmo RL moderno
- ✅ Fácil extensão: copie um ambiente e customize conforme necessário
- ⚠️ Renderização 3D testada apenas em Ubuntu
- ⚠️ GridWorld 3D com renderização requer X11 forwarding em servidores remotos

---

## 💡 Dicas para Extensão

### Adicionar um novo ambiente
1. Crie `gymnasium_env/seu_ambiente.py` herdando de `gym.Env`
2. Implemente `__init__`, `reset()`, `step()` e opcionalmente `render()`
3. Registre com: `gymnasium.register(id='SeuAmbiente-v0', entry_point='...')`
4. Crie um script de treinamento `train_seu_ambiente.py`

### Modificar a função de reward
- Edite o método `step()` de qualquer ambiente
- Teste com um pequeno grid antes de escalar
- Monitore com TensorBoard: `tensorboard --logdir ./log`

### Adicionar observações customizadas
Modifique `_get_obs()` para incluir informações adicionais ao agente (distância, mapa local, etc.)

---

## 📧 Contribuições e Feedback

Contribuições são bem-vindas! Para reportar bugs ou sugerir melhorias, abra uma issue no repositório.

---

**Última atualização**: Abril 2026  
**Status**: ✅ Ativo e mantido
