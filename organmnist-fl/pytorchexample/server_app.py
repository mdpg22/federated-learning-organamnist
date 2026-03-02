"""pytorchexample: A Flower / PyTorch app."""

import json
import torch
from datetime import datetime
from pathlib import Path
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from pytorchexample.task import Net, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Crear directorio de outputs con timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(f"outputs/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGuardando resultados en: {output_dir}")

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Inicializar diccionario para guardar métricas
    all_metrics = {}

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=lambda round, arrays: global_evaluate(
            round, arrays, all_metrics, output_dir
        ),
    )

    # Guardar métricas finales en JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Métricas guardadas en: {metrics_path}")

    # Save final model
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, output_dir / "final_model.pt")


def global_evaluate(
    server_round: int,
    arrays: ArrayRecord,
    all_metrics: dict,
    output_dir: Path
) -> MetricRecord:
    """Evaluate model on central data and save metrics."""
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)

    # Guardar métricas de esta ronda
    all_metrics[server_round] = {
        "accuracy": test_acc,
        "loss": test_loss,
    }
    print(f"[Round {server_round}] accuracy={test_acc:.4f} loss={test_loss:.4f}")

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})