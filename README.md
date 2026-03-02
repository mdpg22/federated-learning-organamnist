# Federated Learning for Medical Image Classification

Final project for the Federated Learning course at Universidad de Navarra (TECNUN).

## Overview
This project implements Federated Learning using the Flower framework to classify
abdominal organ images from the OrganAMNIST dataset (MedMNIST benchmark).
The experiments analyze the impact of data heterogeneity and number of nodes
on federated model performance.

## Dataset
OrganAMNIST (MedMNIST) — 28x28 grayscale images of abdominal CT scans, 11 organ classes.

## Model
Improved CNN with 3 convolutional blocks, Batch Normalization and Dropout.

## Setup
pip install flwr medmnist
pip install -e .

## Run
# Default (IID, 10 nodes, 5 rounds)
flwr run .

# Custom config
flwr run . local-10 --run-config "partitioner-type='dirichlet' alpha=0.1"

# All experiments
bash run_experiments.sh

## Experiments
- IID partitioning with 10, 20 and 50 nodes
- Dirichlet partitioner (alpha=1.0 and alpha=0.1)
- Pathological partitioner (2 and 5 classes per node)

## Results
See report.pdf for the full analysis and plots.
