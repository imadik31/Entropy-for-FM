from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm as std_tqdm
from transformers import HfArgumentParser

from flow_matching.datasets.image_datasets import (
    get_image_dataset,
    get_test_transform,
    get_train_transform,
)
from flow_matching.models import UNetModel
from flow_matching.sampler import PathSampler
from flow_matching.solver import ModelWrapper, ODESolver
from flow_matching.utils import model_size_summary, set_seed

tqdm = partial(std_tqdm, dynamic_ncols=True)


@dataclass
class ScriptArguments:
    do_train: bool = False
    do_sample: bool = False
    dataset: str = "mnist"
    batch_size: int = 128
    n_epochs: int = 10
    learning_rate: float = 1e-3
    sigma_min: float = 0.0
    seed: int = 42
    output_dir: str = "outputs"
    horizontal_flip: bool = False


def train(args: ScriptArguments):
    """Train the flow matching model on the given dataset."""

    output_dir = Path(args.output_dir) / "cfm" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"Using device: {device}")

    # Load the dataset
    dataset = get_image_dataset(
        args.dataset,
        train=True,
        transform=get_train_transform(horizontal_flip=args.horizontal_flip),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Loaded {args.dataset} dataset with {len(dataset):,} samples")

    num_classes = len(dataset.classes)
    input_shape = dataset[0][0].size()
    print(f"{input_shape=}, {num_classes=}")

    # Load the UNet model with class conditioning for flow matching
    flow = UNetModel(
        input_shape,
        num_channels=64,
        num_res_blocks=2,
        num_classes=num_classes,
        class_cond=True,
    ).to(device)
    path_sampler = PathSampler(sigma_min=args.sigma_min)

    # Load the optimizer
    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.learning_rate)
    scaler = GradScaler(enabled=device.type == "cuda")
    print("GradScaler enabled:", scaler._enabled)
    model_size_summary(flow)

    for epoch in range(args.n_epochs):
        flow.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:2d}/{args.n_epochs}")

        for x_1, y in pbar:
            x_1, y = x_1.to(device), y.to(device)

            # Compute the probability path samples
            x_0 = torch.randn_like(x_1)
            t = torch.rand(x_1.size(0), device=device, dtype=x_1.dtype)
            x_t, dx_t = path_sampler.sample(x_0, x_1, t)

            flow.zero_grad(set_to_none=True)

            # Compute the conditional flow matching loss with class conditioning
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                vf_t = flow(t=t, x=x_t, y=y)
                loss = F.mse_loss(vf_t, dx_t)

            # Gradient scaling and backprop
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"loss": loss.item()})

    torch.save(flow.state_dict(), output_dir / "ckpt.pth")
    print(f"Final checkpoint saved to {output_dir / 'ckpt.pth'}")


def generate_samples_and_save_animation(args: ScriptArguments):
    """Generate samples following the flow and save the animation."""

    output_dir = Path(args.output_dir) / "cfm" / args.dataset
    assert output_dir.is_dir(), f"Output directory {output_dir} does not exist"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"Using device: {device}")

    # Load the dataset
    dataset = get_image_dataset(
        args.dataset,
        train=False,
        transform=get_test_transform(),
    )
    input_shape = dataset[0][0].size()
    num_classes = len(dataset.classes)

    # Load the flow model
    flow = UNetModel(
        input_shape,
        num_channels=64,
        num_res_blocks=2,
        num_classes=num_classes,
        class_cond=True,
    ).to(device)
    state_dict = torch.load(output_dir / "ckpt.pth", weights_only=True)
    flow.load_state_dict(state_dict)
    flow.eval()

    # Use ODE solver to sample trajectories
    class WrappedModel(ModelWrapper):
        def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
            return self.model(x=x, t=t, **extras)

    samples_per_class = 10
    sample_steps = 101
    time_steps = torch.linspace(0, 1, sample_steps).to(device)
    class_list = torch.arange(num_classes, device=device).repeat(samples_per_class)

    wrapped_model = WrappedModel(flow)
    step_size = 0.05
    x_init = torch.randn((class_list.size(0), *input_shape), dtype=torch.float32, device=device)
    solver = ODESolver(wrapped_model)
    sol = solver.sample(
        x_init=x_init,
        step_size=step_size,
        method="midpoint",
        time_grid=time_steps,
        return_intermediates=True,
        y=class_list,
    )
    sol = sol.detach().cpu()
    final_samples = sol[-1]

    save_image(final_samples, output_dir / "final_samples.png", nrow=num_classes, normalize=True)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    grid = make_grid(final_samples, nrow=num_classes, normalize=True)
    ax[0].imshow(grid.permute(1, 2, 0))
    ax[0].set_title("Final samples (t = 1.0)", fontsize=16)
    ax[0].axis("off")

    def update(frame: int):
        grid = make_grid(sol[frame], nrow=num_classes, normalize=True)
        ax[1].clear()
        ax[1].imshow(grid.permute(1, 2, 0))
        ax[1].set_title(f"t = {time_steps[frame].item():.2f}", fontsize=16)
        ax[1].axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05, wspace=0.1)
    ani = animation.FuncAnimation(fig, update, frames=sample_steps)
    ani.save(output_dir / "trajectory.gif", writer="pillow", fps=20)
    print(f"Generated trajectory saved to {output_dir / 'trajectory.gif'}")

    # Compute and log entropy estimate
    print("\nComputing entropy estimate...")
    num_entropy_samples = 1000  # Use smaller batch for images to avoid memory issues
    x_init_entropy = torch.randn((num_entropy_samples, *input_shape), dtype=torch.float32, device=device)
    class_list_entropy = torch.arange(num_classes, device=device).repeat(num_entropy_samples // num_classes + 1)
    class_list_entropy = class_list_entropy[:num_entropy_samples]

    _, entropy = solver.sample_with_entropy(
        x_init=x_init_entropy,
        step_size=step_size,
        method="midpoint",
        time_grid=time_steps,
        return_intermediates=False,
        n_probe=2,  # Use 2 probe vectors for Hutchinson estimator
        use_exact_divergence=False,  # Use Hutchinson for images
        y=class_list_entropy,
    )

    # Compute base entropy for reference
    import numpy as np

    flat_dim = np.prod(input_shape)
    base_entropy = 0.5 * flat_dim * (1.0 + np.log(2.0 * np.pi))
    entropy_value = entropy.item()

    # Save to file
    entropy_log_path = output_dir / "entropy_estimate.txt"
    with open(entropy_log_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Entropy Estimation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Image shape: {input_shape}\n")
        f.write(f"Total dimensions: {flat_dim}\n")
        f.write(f"Number of samples: {num_entropy_samples:,}\n")
        f.write(f"Number of time steps: {len(time_steps)}\n")
        f.write(f"Step size: {step_size}\n")
        f.write(f"Integration method: midpoint\n")
        f.write(f"Divergence estimator: Hutchinson (n_probe=2)\n\n")
        f.write(f"Base entropy H(p_0): {base_entropy:.6f} nats\n")
        f.write(f"Estimated entropy H(p_1): {entropy_value:.6f} nats\n")
        f.write(f"Entropy change: {entropy_value - base_entropy:.6f} nats\n\n")
        f.write("=" * 60 + "\n")
        f.write("Formula: H(p_1) = H(p_0) + ∫_0^1 E[∇·v_θ(x_t, t)] dt\n")
        f.write("=" * 60 + "\n")

    print(f"\n{'=' * 60}")
    print("Entropy Estimation Results")
    print("=" * 60)
    print(f"Base entropy H(p_0):       {base_entropy:.6f} nats")
    print(f"Estimated entropy H(p_1):  {entropy_value:.6f} nats")
    print(f"Entropy change:            {entropy_value - base_entropy:+.6f} nats")
    print(f"Results saved to: {entropy_log_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args, *_ = parser.parse_args_into_dataclasses()

    if script_args.do_train:
        train(script_args)

    if script_args.do_sample:
        generate_samples_and_save_animation(script_args)
