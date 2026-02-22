from dataclasses import dataclass, field


@dataclass
class JEMConfig:
    """Configuration for the Joint Energy-Based Model (JEM)."""

    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    sgld_steps: int = 20
    sgld_step_size: float = 1e-3
    sgld_sigma: float = 1e-3
    l2_energy_weight: float = 1e-4
    spectral_norm: bool = True
