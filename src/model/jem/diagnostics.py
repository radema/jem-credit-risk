import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA

from src.model.jem.sampler import SGLDReplayBuffer


class JEMDiagnostics:
    """
    Suite of diagnostic visualizations for the Joint Energy-Based Model (JEM).
    Provides methods to track energy ranges, check OOD density, plot reliability,
    and inspect the SGLD replay buffer.
    """

    @staticmethod
    def plot_energy_ranges(
        e_real_history: list[float],
        e_fake_history: list[float],
        save_path: str | None = None,
    ):
        """
        Plots the average E_real and E_fake trajectories over training steps/epochs.
        This provides a vital thermometer to prevent gradient explosions.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(e_real_history, label="E_real (Data)", color="blue", alpha=0.8)
        plt.plot(e_fake_history, label="E_fake (Buffer)", color="red", alpha=0.8)
        plt.title("Energy Ranges over Training")
        plt.xlabel("Step / Epoch")
        plt.ylabel("Energy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_energy_density(
        e_in_dist: np.ndarray,
        e_out_dist: np.ndarray | None = None,
        save_path: str | None = None,
    ):
        """
        Plots the density histogram of energies for in-distribution vs (optional) out-of-distribution.
        Lower energies correspond to higher likelihood under the model p(x).
        """
        plt.figure(figsize=(10, 6))

        # Plot in-distribution
        plt.hist(
            e_in_dist,
            bins=50,
            alpha=0.6,
            density=True,
            color="blue",
            label="In-Distribution",
        )

        # Plot out-of-distribution (anomalies) if provided
        if e_out_dist is not None:
            plt.hist(
                e_out_dist,
                bins=50,
                alpha=0.6,
                density=True,
                color="red",
                label="Out-of-Distribution",
            )

        plt.title("Energy Density Distribution")
        plt.xlabel("Energy E(x)")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_reliability_diagram(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        num_bins: int = 10,
        save_path: str | None = None,
    ):
        """
        Plots the calibration curve (reliability diagram) to verify if the JEM
        remedies over-confidence.
        """
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=num_bins, strategy="quantile"
        )

        plt.figure(figsize=(8, 8))
        plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="JEM Calibration")
        plt.plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated"
        )

        plt.title("Reliability Diagram")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    @staticmethod
    def inspect_replay_buffer(
        buffer: SGLDReplayBuffer,
        x_real: torch.Tensor,
        save_path: str | None = None,
        n_samples: int = 1000,
    ):
        """
        Inspects the replay buffer against real data by projecting both onto the
        top 2 principal components computed from the real data. Ensures the SGLD
        buffer isn't just pure noise.
        """
        real_np = x_real.detach().cpu().numpy()

        # Validate buffer has something
        if buffer.pointer == 0 and not buffer.is_full:
            print("Buffer is empty, cannot inspect.")
            return

        # Determine actual sample count
        n_inspect = min(
            n_samples, buffer.buffer_size if buffer.is_full else buffer.pointer
        )
        x_fake = buffer.sample(n_inspect)
        fake_np = x_fake.detach().cpu().numpy()

        # Balance sizes for fair visual comparison
        if len(real_np) > n_inspect:
            idx = np.random.choice(len(real_np), n_inspect, replace=False)
            real_np = real_np[idx]

        # Fit PCA on real data only to not let noise dominate the principal components,
        # then project both down to 2 axes.
        pca = PCA(n_components=2)
        real_pca = pca.fit_transform(real_np)
        fake_pca = pca.transform(fake_np)

        plt.figure(figsize=(10, 8))
        plt.scatter(
            real_pca[:, 0],
            real_pca[:, 1],
            alpha=0.5,
            s=15,
            label="Real Data",
            color="blue",
        )
        plt.scatter(
            fake_pca[:, 0],
            fake_pca[:, 1],
            alpha=0.5,
            s=15,
            label="Fake Data (Buffer)",
            color="red",
        )

        plt.title("Replay Buffer Inspection (PCA Projection)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()
