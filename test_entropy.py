"""Quick test script for entropy estimation on existing checkpoints."""

import argparse
import torch
from pathlib import Path
from train_flow_matching_2d import Mlp
from flow_matching.datasets import TOY_DATASETS
from flow_matching import visualization
from flow_matching.solver import ModelWrapper
from torch import Tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="moons")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load dataset
    dataset = TOY_DATASETS[args.dataset](device=device)

    # Load model
    flow = Mlp(dim=dataset.dim, time_dim=1, h=512).to(device)
    flow.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    flow.eval()

    # Wrap model
    class WrappedModel(ModelWrapper):
        def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
            return self.model(x_t=x, t=t)

    wrapped_model = WrappedModel(flow)

    # Test entropy estimation
    print("\nTesting entropy estimation...")
    output_dir = Path(args.checkpoint).parent
    entropy = visualization.compute_and_log_entropy(
        flow=wrapped_model,
        dataset=dataset,
        output_dir=output_dir,
        filename=f"entropy_{args.dataset}_test.txt",
    )

    print(f"\nEntropy estimation completed successfully!")
    print(f"Estimated entropy: {entropy:.6f} nats")

if __name__ == "__main__":
    main()
