"""Simulates a realistic deep learning experiment with meaningful metrics.

Each run produces different training dynamics based on:
- A unique random seed (auto-generated or passed via --seed)
- Hyperparameters (learning_rate, batch_size, weight_decay, dropout, etc.)
- Simulated phenomena: warmup, plateau, lr decay, overfitting, gradient noise

Runs are genuinely different and comparable — varying hyperparameters
shifts loss curves, convergence speed, and final accuracy in realistic ways.
"""

import argparse
import logging
import math
import random
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from polyaxon import settings, tracking

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")

NUM_CLASSES = 10
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# ---------------------------------------------------------------------------
# Hyperparameter-dependent training simulation
# ---------------------------------------------------------------------------


class TrainingSimulator:
    """Generates realistic training curves driven by hyperparameters."""

    def __init__(self, hp, rng):
        self.hp = hp
        self.rng = rng

        # Derive "true" convergence characteristics from hyperparameters
        # Higher lr -> faster early drop but risk of instability
        lr = hp.learning_rate
        self.lr_factor = min(lr / 0.001, 3.0)

        # Larger batch -> smoother curves but potentially worse generalisation
        self.batch_noise = 0.04 * (32 / hp.batch_size)

        # Weight decay controls overfitting gap
        self.overfit_rate = max(0.0, 0.12 - hp.weight_decay * 40)

        # Dropout reduces overfitting but slows training
        self.dropout_penalty = hp.dropout * 0.3

        # Model capacity (width multiplier) affects ceiling accuracy
        self.capacity = hp.model_width / 64.0
        self.ceiling_acc = min(
            0.96, 0.75 + 0.15 * self.capacity - self.dropout_penalty * 0.1
        )

        # Final achievable loss floor
        self.loss_floor = max(
            0.08, 0.5 - 0.3 * self.capacity + self.dropout_penalty * 0.2
        )

        # Convergence speed (epochs to reach ~63% of improvement)
        self.tau = max(5, 30 / self.lr_factor - hp.model_width * 0.05)

        # Random per-run personality
        self.plateau_epoch = rng.randint(int(hp.epochs * 0.25), int(hp.epochs * 0.55))
        self.plateau_len = rng.randint(3, max(4, int(hp.epochs * 0.1)))
        self.spike_epoch = (
            rng.choice([None, None, rng.randint(5, hp.epochs - 5)])
            if hp.epochs >= 10
            else None
        )
        self.initial_loss = 2.3 + rng.uniform(-0.1, 0.1)  # ~ log(10) for 10 classes

    def train_loss(self, epoch):
        t = epoch / self.tau
        base = self.loss_floor + (self.initial_loss - self.loss_floor) * math.exp(-t)

        # Plateau effect
        if self.plateau_epoch <= epoch < self.plateau_epoch + self.plateau_len:
            base *= 1.0 + 0.05 * math.sin((epoch - self.plateau_epoch) * 0.5)

        # Learning rate spike / instability for high lr
        if self.spike_epoch and epoch == self.spike_epoch:
            base *= 1.0 + self.lr_factor * 0.3

        # Gradient noise
        noise = self.rng.gauss(0, self.batch_noise * base)
        return max(0.01, base + noise)

    def val_loss(self, epoch, train_loss):
        # Validation tracks training but with overfitting gap that grows
        overfit_gap = self.overfit_rate * (epoch / self.hp.epochs) ** 1.5
        val_noise = self.rng.gauss(0, self.batch_noise * 1.5 * train_loss)
        return max(0.01, train_loss * (1.0 + overfit_gap) + val_noise)

    def accuracy_from_loss(self, loss, is_val=False):
        # Invert cross-entropy-ish loss to accuracy
        acc = max(0.0, 1.0 - loss / self.initial_loss)
        acc = min(self.ceiling_acc, acc)
        if is_val:
            acc -= self.rng.uniform(0, 0.02)
        return max(0.0, acc)

    def learning_rate_at(self, epoch):
        lr = self.hp.learning_rate
        # Warmup
        if epoch < self.hp.warmup_epochs:
            return lr * (epoch + 1) / self.hp.warmup_epochs
        # Cosine decay after warmup
        progress = (epoch - self.hp.warmup_epochs) / max(
            1, self.hp.epochs - self.hp.warmup_epochs
        )
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def gradient_norm(self, epoch, train_loss):
        base = train_loss * 2.0 * self.lr_factor
        noise = self.rng.gauss(0, 0.1 * base)
        if self.spike_epoch and epoch == self.spike_epoch:
            base *= 3.0
        return max(0.01, base + noise)


# ---------------------------------------------------------------------------
# Per-class metrics (confusion matrix, precision/recall, ROC)
# ---------------------------------------------------------------------------


def generate_confusion_matrix(epoch, sim, rng):
    """Generates a confusion matrix that improves over training."""
    acc = sim.accuracy_from_loss(sim.train_loss(epoch))
    n_samples = 200
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=float)

    for true_cls in range(NUM_CLASSES):
        for _ in range(n_samples // NUM_CLASSES):
            if rng.random() < acc:
                cm[true_cls, true_cls] += 1
            else:
                # Distribute errors — some classes are more confusable
                confuse_weights = np.ones(NUM_CLASSES)
                confuse_weights[true_cls] = 0
                # Neighboring classes are more confusable
                for c in range(NUM_CLASSES):
                    if abs(c - true_cls) == 1:
                        confuse_weights[c] = 3.0
                confuse_weights /= confuse_weights.sum()
                pred = rng.choices(range(NUM_CLASSES), weights=confuse_weights)[0]
                cm[true_cls, pred] += 1

    return cm


def trapezoidal_area(y, x):
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(y, x))


def generate_roc_curve(epoch, sim, rng):
    """Generates ROC data that improves as the model trains."""
    acc = sim.accuracy_from_loss(sim.train_loss(epoch))
    # Simulate scores for positive and negative examples
    n = 500
    pos_scores = np.clip(
        rng.gauss(acc, 0.2) + np.array([rng.gauss(0, 0.15) for _ in range(n)]), 0, 1
    )
    neg_scores = np.clip(
        rng.gauss(1 - acc, 0.25) + np.array([rng.gauss(0, 0.15) for _ in range(n)]),
        0,
        1,
    )

    y_true = np.concatenate([np.ones(n), np.zeros(n)])
    y_score = np.concatenate([pos_scores, neg_scores])

    # Compute ROC
    thresholds = np.linspace(1, 0, 200)
    tpr, fpr = [], []
    for t in thresholds:
        preds = (y_score >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        tn = ((preds == 0) & (y_true == 0)).sum()
        tpr.append(tp / max(1, tp + fn))
        fpr.append(fp / max(1, fp + tn))

    # AUC via trapezoidal rule
    auc = trapezoidal_area(tpr, fpr)
    return np.array(fpr), np.array(tpr), abs(auc)


def generate_pr_curve(epoch, sim, rng):
    """Generates PR curve that improves over training."""
    acc = sim.accuracy_from_loss(sim.train_loss(epoch))
    n = 500
    pos_scores = np.clip(acc + np.array([rng.gauss(0, 0.2) for _ in range(n)]), 0, 1)
    neg_scores = np.clip(
        (1 - acc) * 0.6 + np.array([rng.gauss(0, 0.2) for _ in range(n)]), 0, 1
    )

    y_true = np.concatenate([np.ones(n), np.zeros(n)])
    y_score = np.concatenate([pos_scores, neg_scores])

    thresholds = np.linspace(1, 0, 200)
    precision_pts, recall_pts = [], []
    for t in thresholds:
        preds = (y_score >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        precision_pts.append(prec)
        recall_pts.append(rec)

    ap = trapezoidal_area(precision_pts, recall_pts)
    return np.array(precision_pts), np.array(recall_pts), abs(ap)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def plot_weight_distributions(epoch, sim, rng):
    """Simulates weight distributions that sharpen during training."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    layer_names = ["conv1", "conv3", "fc1"]
    for ax, name in zip(axes, layer_names):
        # Weights start ~N(0, 0.1) and shift as training progresses
        scale = 0.1 + 0.05 * (epoch / sim.hp.epochs)
        shift = rng.gauss(0, 0.02 * epoch / sim.hp.epochs)
        weights = np.array([rng.gauss(shift, scale) for _ in range(2000)])
        ax.hist(weights, bins=50, alpha=0.7, color="steelblue", edgecolor="none")
        ax.set_title(f"{name} weights")
        ax.set_xlim(-0.5, 0.5)
    fig.suptitle(f"Weight distributions — epoch {epoch}")
    fig.tight_layout()
    return fig


def plot_gradient_flow(epoch, sim, rng):
    """Bar chart of per-layer gradient magnitudes."""
    layers = ["conv1", "bn1", "conv2", "bn2", "conv3", "bn3", "fc1", "fc2"]
    grad_base = sim.gradient_norm(epoch, sim.train_loss(epoch))

    # Gradients decay in earlier layers (vanishing gradient effect)
    grads = []
    for i, name in enumerate(layers):
        depth_factor = 0.3 + 0.7 * (i / len(layers))
        g = grad_base * depth_factor * (1 + rng.gauss(0, 0.15))
        grads.append(max(1e-5, g))

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#d32f2f" if g > grad_base * 1.5 else "#1976d2" for g in grads]
    ax.barh(layers, grads, color=colors)
    ax.set_xlabel("Gradient magnitude")
    ax.set_title(f"Gradient flow — epoch {epoch}")
    fig.tight_layout()
    return fig


def plot_feature_maps(epoch, sim, rng):
    """Simulates feature map activations that become more structured over training."""
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for ax in axes.flat:
        structure = min(1.0, epoch / (sim.hp.epochs * 0.6))
        # Early: random noise. Late: structured patterns (edges, blobs)
        noise = rng.random() * (1 - structure)
        base = np.array([[rng.gauss(0, 1) for _ in range(8)] for _ in range(8)])
        # Add spatial structure
        x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8))
        freq = rng.uniform(1, 4)
        phase = rng.uniform(0, 2 * math.pi)
        pattern = np.sin(freq * x_grid + phase) * np.cos(freq * y_grid + phase)
        activation = noise * base + structure * pattern
        ax.imshow(activation, cmap="viridis", interpolation="nearest")
        ax.axis("off")
    fig.suptitle(f"Feature activations — epoch {epoch}")
    fig.tight_layout()
    return fig


def plot_loss_landscape(epoch, sim, rng):
    """2D loss landscape cross-section around current parameters."""
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    # As training progresses, the landscape narrows (closer to minimum)
    sharpness = 0.5 + 2.0 * (epoch / sim.hp.epochs)
    offset_x = rng.gauss(0, 0.3)
    offset_y = rng.gauss(0, 0.3)
    Z = (
        sharpness * (X - offset_x) ** 2
        + sharpness * 0.7 * (Y - offset_y) ** 2
        + 0.3 * np.sin(3 * X) * np.cos(3 * Y) * max(0.1, 1 - epoch / sim.hp.epochs)
    )

    ax.contourf(X, Y, Z, levels=30, cmap="RdYlBu_r")
    ax.plot(offset_x, offset_y, "k*", markersize=15, label="current params")
    ax.set_xlabel("param direction 1")
    ax.set_ylabel("param direction 2")
    ax.set_title(f"Loss landscape — epoch {epoch}")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def init_tracking(args):
    if settings.CLIENT_CONFIG.is_managed:
        return tracking.init()
    return tracking.init(
        project=args.project,
        name=args.run_name,
        tags=["sim"],
    )


def main(args):
    seed = args.seed if args.seed is not None else random.randint(0, 2**31)
    rng = random.Random(seed)
    np.random.seed(seed % (2**32))

    logger.info("Run seed: %d", seed)
    logger.info(
        "Hyperparameters: lr=%.5f, batch_size=%d, weight_decay=%.4f, "
        "dropout=%.2f, model_width=%d, warmup=%d, epochs=%d",
        args.learning_rate,
        args.batch_size,
        args.weight_decay,
        args.dropout,
        args.model_width,
        args.warmup_epochs,
        args.epochs,
    )

    sim = TrainingSimulator(args, rng)

    run = init_tracking(args)

    # Log hyperparameters
    tracking.log_inputs(
        seed=seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        model_width=args.model_width,
        warmup_epochs=args.warmup_epochs,
        optimizer=args.optimizer,
        scheduler="cosine",
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        t_start = time.time()

        # --- Core metrics ---
        train_loss = sim.train_loss(epoch)
        val_loss = sim.val_loss(epoch, train_loss)
        train_acc = sim.accuracy_from_loss(train_loss)
        val_acc = sim.accuracy_from_loss(val_loss, is_val=True)
        lr = sim.learning_rate_at(epoch)
        grad_norm = sim.gradient_norm(epoch, train_loss)

        # Track best
        best_val_loss = min(best_val_loss, val_loss)
        best_val_acc = max(best_val_acc, val_acc)

        # Per-class metrics (top-1 accuracy per class, simulated)
        per_class_acc = {}
        for i, name in enumerate(CLASS_NAMES):
            class_offset = rng.gauss(0, 0.05)
            per_class_acc[name] = max(0.0, min(1.0, train_acc + class_offset))

        # Log scalars
        tracking.log_metrics(
            step=epoch,
            train_loss=round(train_loss, 5),
            val_loss=round(val_loss, 5),
            train_accuracy=round(train_acc, 5),
            val_accuracy=round(val_acc, 5),
            learning_rate=round(lr, 7),
            gradient_norm=round(grad_norm, 5),
            epoch_time=round(time.time() - t_start + rng.uniform(0.5, 2.0), 3),
            overfit_gap=round(val_loss - train_loss, 5),
        )

        # Log per-class accuracy
        for cls_name, cls_acc in per_class_acc.items():
            tracking.log_metric(
                name=f"class_acc/{cls_name}",
                value=round(cls_acc, 4),
                step=epoch,
            )

        # Progress
        tracking.log_progress((epoch + 1) / args.epochs)

        # --- Weight / gradient distributions (every N epochs) ---
        if epoch % max(1, args.epochs // 10) == 0:
            # Weight histogram
            for layer_idx, layer_name in enumerate(["conv1", "conv3", "fc1"]):
                scale = 0.1 + 0.05 * (epoch / args.epochs)
                shift = rng.gauss(0, 0.02 * epoch / args.epochs)
                weights = np.array([rng.gauss(shift, scale) for _ in range(2000)])
                tracking.log_histogram(
                    f"weights/{layer_name}", weights, "auto", step=epoch
                )

            # Gradient histogram
            for layer_name in ["conv1", "fc1", "fc2"]:
                grad_scale = grad_norm * rng.uniform(0.5, 1.5)
                grads = np.array([rng.gauss(0, grad_scale * 0.1) for _ in range(2000)])
                tracking.log_histogram(
                    f"gradients/{layer_name}", grads, "auto", step=epoch
                )

        # --- Charts (at key checkpoints) ---
        log_interval = max(1, args.epochs // 8)
        if epoch % log_interval == 0 or epoch == args.epochs - 1:
            # Confusion matrix
            cm = generate_confusion_matrix(epoch, sim, rng)
            tracking.log_confusion_matrix(
                "confusion_matrix",
                x=CLASS_NAMES,
                y=CLASS_NAMES,
                z=cm.tolist(),
                step=epoch,
            )

            # ROC curve
            fpr, tpr, auc_val = generate_roc_curve(epoch, sim, rng)
            tracking.log_roc_auc_curve(
                name="roc_curve", fpr=fpr, tpr=tpr, auc=auc_val, step=epoch
            )

            # PR curve
            prec, rec, ap = generate_pr_curve(epoch, sim, rng)
            tracking.log_pr_curve(
                name="pr_curve",
                precision=prec,
                recall=rec,
                average_precision=ap,
                step=epoch,
            )

            # Matplotlib figures
            fig = plot_weight_distributions(epoch, sim, rng)
            tracking.log_mpl_image(fig, "weight_distributions", step=epoch)
            plt.close(fig)

            fig = plot_gradient_flow(epoch, sim, rng)
            tracking.log_mpl_image(fig, "gradient_flow", step=epoch)
            plt.close(fig)

            fig = plot_feature_maps(epoch, sim, rng)
            tracking.log_mpl_image(fig, "feature_activations", step=epoch)
            plt.close(fig)

            fig = plot_loss_landscape(epoch, sim, rng)
            tracking.log_mpl_image(fig, "loss_landscape", step=epoch)
            plt.close(fig)

            # Training summary text
            tracking.log_text(
                "training_log",
                text=(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                    f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} | "
                    f"lr={lr:.6f} grad_norm={grad_norm:.4f} | "
                    f"best_val_loss={best_val_loss:.4f}"
                ),
                step=epoch,
            )

        logger.info(
            "Epoch %3d/%d — train_loss=%.4f val_loss=%.4f train_acc=%.4f val_acc=%.4f lr=%.6f",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
            lr,
        )

    # --- Final outputs ---
    tracking.log_outputs(
        final_train_loss=round(train_loss, 5),
        final_val_loss=round(val_loss, 5),
        final_val_accuracy=round(val_acc, 5),
        best_val_loss=round(best_val_loss, 5),
        best_val_accuracy=round(best_val_acc, 5),
        total_epochs=args.epochs,
        seed=seed,
    )

    logger.info(
        "Training complete. Final val_loss=%.4f val_acc=%.4f; "
        "best val_loss=%.4f best val_acc=%.4f",
        val_loss,
        val_acc,
        best_val_loss,
        best_val_acc,
    )
    tracking.end()
    print(f"Polyaxon run ID: {run.run_uuid}")


def validate_args(parser, args):
    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than zero")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be zero or greater")
    if not 0 <= args.dropout <= 1:
        parser.error("--dropout must be between zero and one")
    if args.model_width <= 0:
        parser.error("--model-width must be greater than zero")
    if not 0 <= args.warmup_epochs <= args.epochs:
        parser.error("--warmup-epochs must be between zero and --epochs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a deep learning experiment")
    parser.add_argument(
        "--project",
        type=str,
        default="quick-start",
        help="Polyaxon project name or OWNER/PROJECT",
    )
    parser.add_argument(
        "--run-name",
        "--run_name",
        dest="run_name",
        type=str,
        default=None,
        help="Name assigned to the tracked run",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--learning-rate",
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--weight-decay",
        "--weight_decay",
        dest="weight_decay",
        type=float,
        default=0.0001,
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--model-width",
        "--model_width",
        dest="model_width",
        type=int,
        default=64,
        help="Base channel width / hidden size multiplier",
    )
    parser.add_argument(
        "--warmup-epochs",
        "--warmup_epochs",
        dest="warmup_epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw"]
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (auto-generated if omitted)"
    )
    args = parser.parse_args()
    validate_args(parser, args)
    main(args)

# Created simulate_dl_experiment.py. Here's what it does differently from all.py:

#   Realistic training dynamics:
#   - Loss follows exponential decay with a proper floor, not 10/(step+1)
#   - Simulates warmup, cosine LR decay, plateaus, and occasional gradient spikes
#   - Overfitting gap grows naturally over time (controlled by weight_decay and
#   dropout)
#   - Gradient norms decay with training and exhibit vanishing gradients in earlier
#    layers

#   Every run is different:
#   - Auto-generates a random seed if --seed is not passed, so each invocation
#   produces unique curves
#   - Per-run "personality" — random plateau timing, optional loss spikes, noise
#   levels

#   Hyperparameters actually matter:
#   - --learning_rate affects convergence speed and instability risk
#   - --batch_size controls gradient noise (smaller = noisier curves)
#   - --weight_decay / --dropout control the train-val gap (overfitting)
#   - --model_width affects ceiling accuracy and convergence
#   - --optimizer is logged for comparison

#   Richer logged artifacts:
#   - Per-class accuracy for 10 CIFAR-like classes
#   - Confusion matrices that improve as training progresses
#   - ROC and PR curves computed from simulated score distributions
#   - Weight and gradient histograms per layer
#   - Feature activation maps that transition from noise to structure
#   - Loss landscape cross-sections that sharpen over training
#   - Gradient flow bar charts showing per-layer magnitudes

#   Usage examples:
#   # Default run (unique each time)
#   python simulate_dl_experiment.py

#   # High LR, small batch — expect noisy curves, fast convergence
#   python simulate_dl_experiment.py --learning_rate 0.01 --batch_size 16

#   # Strong regularization — small overfit gap, slower training
#   python simulate_dl_experiment.py --weight_decay 0.01 --dropout 0.5

#   # Reproducible run
#   python simulate_dl_experiment.py --seed 42
