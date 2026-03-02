#!/bin/bash

echo "🚀 Iniciando todos los experimentos..."

echo "📊 Experimento 1: IID 10 nodos"
flwr run . local-10

echo "📊 Experimento 2: IID 20 nodos"
flwr run . local-20

echo "📊 Experimento 3: IID 50 nodos"
flwr run . local-50

echo "�� Experimento 4: Dirichlet alpha=1.0"
flwr run . local-10 --run-config "partitioner-type='dirichlet' alpha=1.0"

echo "📊 Experimento 5: Dirichlet alpha=0.1"
flwr run . local-10 --run-config "partitioner-type='dirichlet' alpha=0.1"

echo "📊 Experimento 6: Pathological 2 clases"
flwr run . local-10 --run-config "partitioner-type='pathological' num-classes-per-partition=2"

echo "📊 Experimento 7: Pathological 5 clases"
flwr run . local-10 --run-config "partitioner-type='pathological' num-classes-per-partition=5"

echo "✅ Todos los experimentos completados!"
