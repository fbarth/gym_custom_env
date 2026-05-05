import pytest
from aps07.configs import (
    ConfigName,
    GridSize,
    PHASE_TIMESTEPS,
    PHASE_MAX_STEPS,
    PHASE_OBSTACLES,
    SEEDS,
    get_phase_n_envs,
)


def test_seeds_are_three():
    assert SEEDS == (0, 1, 2)


def test_grid_sizes_have_timesteps():
    for size in (5, 10, 20):
        assert size in PHASE_TIMESTEPS


def test_grid_sizes_have_obstacles():
    assert PHASE_OBSTACLES == {5: 3, 10: 12, 20: 48}


def test_grid_sizes_have_max_steps():
    assert PHASE_MAX_STEPS == {5: 200, 10: 500, 20: 1000}


def test_n_envs_for_recurrent_is_two_in_all_grids():
    for size in (5, 10, 20):
        assert get_phase_n_envs("curriculum_recurrent", size) == 2


def test_n_envs_for_ppo_is_four_for_small_grids_two_for_large():
    assert get_phase_n_envs("baseline", 5) == 4
    assert get_phase_n_envs("baseline", 10) == 4
    assert get_phase_n_envs("baseline", 20) == 2


def test_config_names_complete():
    expected = {"baseline", "curriculum", "curriculum_enriched", "curriculum_recurrent"}
    assert set(ConfigName.__args__) == expected
