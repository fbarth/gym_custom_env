# Coverage Path Planning (CPP) Implementation

## Sumário das Mudanças

Este documento descreve as mudanças implementadas para adicionar suporte a problemas de Coverage Path Planning usando o framework Gymnasium.

## Arquivos Adicionados

### 1. `gymnasium_env/grid_world_cpp.py`
- **Descrição**: Novo ambiente `GridWorldCPPEnv` especializado em Coverage Path Planning
- **Características principais**:
  - Rastreamento de células visitadas em uma grade quadrada
  - Renderização com pygame mostrando células visitadas
  - Função de reward baseada em cobertura do ambiente
  - Suporte a observações em formato dicionário e flattenado
  - Configuração de tamanho do grid e número máximo de passos

### 2. `run_grid_world_cpp.py`
- **Descrição**: Script de teste do ambiente com agente aleatório e renderização
- **Funcionalidade**: Executa o ambiente com ações aleatórias e visualiza o desempenho
- **Uso**: `python run_grid_world_cpp.py`

### 3. `test_grid_world_cpp.py`
- **Descrição**: Script de validação do ambiente em modo headless
- **Funcionalidade**: Testa as funcionalidades principais do ambiente sem renderização
- **Uso**: `python test_grid_world_cpp.py`

### 4. `train_grid_world_cpp.py`
- **Descrição**: Script para treinar, testar e executar agentes no ambiente CPP using PPO
- **Funcionalidades**:
  - `train`: Treina um novo agente usando PPO
  - `test`: Testa um agente treinado em múltiplos episódios
  - `run`: Executa um episódio demonstrativo com renderização
- **Uso**: 
  - `python train_grid_world_cpp.py train` - Treina modelo
  - `python train_grid_world_cpp.py test` - Testa modelo treinado
  - `python train_grid_world_cpp.py run` - Executa demonstração

### 5. `CHANGELOG_CPP.md`
- **Descrição**: Este arquivo
- **Propósito**: Documentar as mudanças e implementações

## Arquivos Modificados

### `README.md`
- **Mudança**: Substituição da seção "Uso do ambiente GridWorld para problemas de Coverage Path Planning"
- **Adição**: Documentação completa sobre:
  - O que é Coverage Path Planning
  - Diferenças na função de reward (antiga vs. nova)
  - Como usar o novo ambiente
  - Inspiração teórica de trabalhos acadêmicos

## Função de Reward para CPP

A função de reward foi implementada para encorajar a cobertura máxima do ambiente:

```python
reward = -0.1  # Custo de cada passo

# Recompensa por célula nova visitada
if célula_é_nova:
    reward += 1.0
else:
    # Penalidade por revisitar célula
    reward -= 1.0

# Bônus por cobertura completa (100%)
if cobertura == 100%:
    reward += 100.0
```

### Justificativa das Componentes:

- **-0.1 (custo de passo)**: Incentiva soluções eficientes em termos de número de passos
- **+1.0 (célula nova)**: Recompensa exploração de novas áreas
- **-1.0 (célula revisitada)**: Penaliza comportamento aleatório e rerevisitas
- **+100.0 (cobertura 100%)**: Objetivo principal - altamente recompensador

## Validação

O ambiente foi testado com:

1. **Teste de criação e inicialização**: Verificação de estrutura e observação
2. **Teste de múltiplos episódios**: Consistência e variabilidade
3. **Teste de agente aleatório em grid 5x5**: Atingiu cobertura média de ~64.8% em 50 passos

## Resultados de Teste

```
Test 1: Environment Creation ✓
Test 2: Environment Reset ✓
Test 3: Single Episode (5x5 grid, 50 steps)
  - Final coverage: 76.0%
  - Cells visited: 19/25
  - Total reward: -19.00

Test 4: Five Episodes (5x5 grid)
  - Average coverage: 64.8%
```

## Próximas Etapas Sugeridas

1. Treinar agentes usando PPO no grid 5x5 até convergência
2. Testar escalabilidade em grids maiores (10x10, 20x20)
3. Implementar obstáculos no ambiente CPP
4. Comparar diferentes funções de reward
5. Aplicar em problemas reais de robótica/otimização

## Referências Acadêmicas

As seguintes referências acadêmicas inspiraram a implementação:

- "A Deep Reinforcement Learning Approach for the Patrolling Problem of Water Resources Through Autonomous Surface Vehicles: The Ypacarai Lake Case"
- "A Comprehensive Survey on Coverage Path Planning for Mobile Robots in Dynamic Environments"

## Notas de Implementação

- O ambiente utiliza a biblioteca pygame para renderização
- Suporta tanto renderização "human" quanto "rgb_array" para logging de videos
- Observações podem ser obtidas em formato dicionário ou flattenadas (compatível com algoritmos padrão)
- Hiperparâmetros ajustáveis: tamanho da grid, número máximo de passos, Taxa de aprendizado, coeficiente de entropia
