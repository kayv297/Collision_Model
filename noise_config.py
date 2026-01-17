"""
Centralized noise model configuration for collision model simulations.
Allows easy switching between different noise models for analysis.
"""
from enum import Enum

class NoiseModel(Enum):
    """Available noise models for quantum simulation."""
    NONE = "none"
    FAILED_INTERACTIONS = "failed"
    DEPHASING = "dephasing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    DEPOLARIZING = "depolarizing"

class NoiseConfig:
    """
    Container for noise model parameters.
    
    Note: seed is only used for failed_interactions (classical control failure).
    Quantum noise channels (dephasing, amplitude damping, depolarizing) are
    deterministic ensemble-averaged operations that don't require seeds.
    
    Attributes:
        model: NoiseModel enum specifying which noise to apply
        shuttle_fail_prob: Probability of shuttle-register interaction failure
        dephasing_prob: Probability parameter for dephasing channel
        amplitude_damping_gamma: Decay parameter for amplitude damping
        depolarizing_prob: Probability parameter for depolarizing channel
        seed: Random seed for reproducibility (only used for failed_interactions)
    """
    
    def __init__(self, 
                 model=NoiseModel.NONE,
                 shuttle_fail_prob=0.0,
                 dephasing_prob=0.0,
                 amplitude_damping_gamma=0.0,
                 depolarizing_prob=0.0,
                 seed=None):
        self.model = model
        self.shuttle_fail_prob = shuttle_fail_prob
        self.dephasing_prob = dephasing_prob
        self.amplitude_damping_gamma = amplitude_damping_gamma
        self.depolarizing_prob = depolarizing_prob
        self.seed = seed  # Only affects failed_interactions
        
    def requires_density_matrix(self):
        """Check if this noise model requires density matrix representation."""
        return self.model in [
            NoiseModel.DEPHASING,
            NoiseModel.AMPLITUDE_DAMPING,
            NoiseModel.DEPOLARIZING
        ]
    
    def get_channel_params(self):
        """
        Get parameters for quantum noise channels.
        These are applied deterministically at every shuttle-register interaction.
        
        Returns:
            dict: Parameters to apply to collision dynamics
        """
        return {
            'dephasing_prob': self.dephasing_prob if self.model == NoiseModel.DEPHASING else 0.0,
            'amplitude_damping_gamma': self.amplitude_damping_gamma if self.model == NoiseModel.AMPLITUDE_DAMPING else 0.0,
            'depolarizing_prob': self.depolarizing_prob if self.model == NoiseModel.DEPOLARIZING else 0.0,
        }
    
    def get_filename_suffix(self):
        """Generate filename suffix for this noise configuration."""
        if self.model == NoiseModel.NONE:
            return ""
        elif self.model == NoiseModel.FAILED_INTERACTIONS:
            return f"_fail{int(self.shuttle_fail_prob*100):02d}"
        elif self.model == NoiseModel.DEPHASING:
            base = f"_deph{int(self.dephasing_prob*100):02d}"
            return base
        elif self.model == NoiseModel.AMPLITUDE_DAMPING:
            base = f"_ampdamp{int(self.amplitude_damping_gamma*100):02d}"
            return base
        elif self.model == NoiseModel.DEPOLARIZING:
            base = f"_depol{int(self.depolarizing_prob*100):02d}"
            return base
    
    def __str__(self):
        """Human-readable description."""
        if self.model == NoiseModel.NONE:
            return "No noise (pure state)"
        
        parts = [f"Noise model: {self.model.value}"]
        
        if self.shuttle_fail_prob > 0:
            parts.append(f"  Shuttle failure: {self.shuttle_fail_prob:.2%} (seed={self.seed})")
        if self.dephasing_prob > 0:
            parts.append(f"  Dephasing: p={self.dephasing_prob:.4f} (deterministic channel)")
        if self.amplitude_damping_gamma > 0:
            parts.append(f"  Amplitude damping: γ={self.amplitude_damping_gamma:.4f} (deterministic channel)")
        if self.depolarizing_prob > 0:
            parts.append(f"  Depolarizing: p={self.depolarizing_prob:.4f} (deterministic channel)")
        
        return "\n".join(parts)


# ============================================================================
# PREDEFINED CONFIGURATIONS FOR EASY SWITCHING
# ============================================================================

# No noise baseline
NO_NOISE = NoiseConfig(model=NoiseModel.NONE, shuttle_fail_prob=0.0)

# Failed interactions only (5%) - needs seed for reproducibility
FAILED = NoiseConfig(
    model=NoiseModel.FAILED_INTERACTIONS,
    shuttle_fail_prob=0.05,
    seed=42
)

# Quantum noise channels (no seed needed - deterministic!)
DEPHASING = NoiseConfig(
    model=NoiseModel.DEPHASING,
    dephasing_prob=0.1
)

AMPLITUDE_DAMPING = NoiseConfig(
    model=NoiseModel.AMPLITUDE_DAMPING,
    amplitude_damping_gamma=0.01
)

DEPOLARIZING = NoiseConfig(
    model=NoiseModel.DEPOLARIZING,
    depolarizing_prob=0.01
)