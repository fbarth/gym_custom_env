## Arquivos Adicionados e Suas Respectivas Mudanças em Relação aos Originais

**train_grid_world_cpp:**

Sumário das mudanças:

- Substituição do algoritmo PPO → MaskablePPO
- Implementação de máscara de ações para evitar movimentos inválidos
- Ajuste de hiperparâmetros (gamma, learning rate, batch size, etc.)
- Aumento do número de timesteps de treinamento
- Uso de uma rede neural maior (arquitetura customizada)
- Adição de avaliação durante o treinamento e salvamento do melhor modelo
- Inclusão de modo random como baseline
- Mudança da execução de política determinística → estocástica

**grid_world_obstacles_cpp:**

Sumário das mudanças:

- Alteração do objetivo: de alcançar um alvo → explorar completamente o ambiente (Coverage Path Planning)
- Alteração do espaço de observação: de informação local (vizinhança) → mapa completo do grid (full observability)
- Implementação de controle de: células visitadas, cobertura total do ambiente, Nova função de reward baseada em, exploração, eficiência, penalização de loops
- Implementação de máscara de ações válidas
- Melhoria da renderização: células visitadas destacadas, exibição de porcentagem de cobertura

## Explicando alterações na nova função de reward

```python
reward = -0.1  # Custo de movimento
if (not in_bounds) or hits_obstacle:
            # Illegal move — agent stays in place
            reward = -1.0
            self.steps_since_new += 1
        else:
            self._agent_location = proposed.copy()
            cell = (int(proposed[0]), int(proposed[1]))

            if not self.visited[cell]:
                # ✓ New cell — primary positive signal
                self.visited[cell] = True
                reward = 1.0
                self.steps_since_new = 0
            else:
                # mild penalty; agent should be free to use visited
                # cells as corridors without being heavily punished for it.
                reward = -0.2
                self.steps_since_new += 1

        self.count_steps += 1

        # ── Terminal condition ──────────────────────────────────────────────
        terminated = bool(self.visited.sum() >= self.total_free_cells)
        if terminated:
            reward += 30.0      #completion bonus

        # ── Per-step time penalty ───────────────────────────────────────────
        reward -= 0.02

        # ── Stagnation/Anti-Loop penalty ──────────────────────────────────────────────
        # Kicks in after 7 consecutive steps without discovering a new cell.
        if self.steps_since_new >= 7 and not terminated:
            reward -= 0.5

        # ── Truncation ──────────────────────────────────────────────────────
        truncated = bool(self.count_steps >= self.max_steps and not terminated)
        if truncated:
            # -5 flat penalty + partial coverage credit
            reward -= 5.0
            reward += 10.0 * self._coverage_ratio()
```

A função de reward foi projetada para equilibrar exploração, eficiência e estabilidade do aprendizado:

- Movimentos inválidos (-1.0): penalizam ações impossíveis (colisões com parede/obstáculo)
- Revisita (-0.2): penaliza levemente revisitar células permitindo seu uso como “corredores”
- Nova célula (+1.0): principal incentivo para exploração
- Penalidade por passo (-0.02): incentiva soluções mais eficientes
- Penalidade por estagnação (-0.5): aplicada após vários passos sem progresso, evitando loops
- Bônus de conclusão (+30): forte incentivo para cobertura completa
- Truncamento: penalidade fixa + reward proporcional à cobertura, garantindo aprendizado mesmo sem solução completa

## Referências Acadêmicas

Fez-se uso das referências acadêmicas abaixo na implementação:

- "A Deep Reinforcement Learning Approach for the Patrolling Problem of Water Resources Through Autonomous Surface Vehicles: The Ypacarai Lake Case"
- "A Comprehensive Survey on Coverage Path Planning for Mobile Robots in Dynamic Environments"