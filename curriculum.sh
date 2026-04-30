#!bin/bash

# Treinando agentes para ambiente 5x5 com 3 obstáculos e 200 passos máximos
python train_grid_world_cpp.py train 5 3 200 500000 &
python train_grid_world_cpp.py train 5 3 200 500000 &
python train_grid_world_cpp.py train 5 3 200 500000 &
python train_grid_world_cpp.py train 5 3 200 500000 &
python train_grid_world_cpp.py train 5 3 200 500000 &

# Treinando agentes para ambiente 10x10 com 12 obstáculos e 500 passos máximos
python train_grid_world_cpp.py train 10 12 500 500000 &
python train_grid_world_cpp.py train 10 12 500 500000 &
python train_grid_world_cpp.py train 10 12 500 500000 &
python train_grid_world_cpp.py train 10 12 500 500000 &
python train_grid_world_cpp.py train 10 12 500 500000 &

# Treinando agentes para ambiente 20x20 com 48 obstáculos e 1000 passos máximos
python train_grid_world_cpp.py train 20 48 1000 500000 &
python train_grid_world_cpp.py train 20 48 1000 500000 &
python train_grid_world_cpp.py train 20 48 1000 500000 &
python train_grid_world_cpp.py train 20 48 1000 500000 &
python train_grid_world_cpp.py train 20 48 1000 500000 &


