# Technical Documentation Index

## Hierarchical Computational Modeling of Vipassana Meditation

This documentation provides comprehensive technical details for the Active Inference-based meditation simulation framework. The system models expert-novice differences in focused attention meditation through a three-level hierarchical architecture.

---

## Core Implementation

### [Active Inference Core](./active_inference_core.md)
- **Primary Module**: `act_inf_learner.py`
- **Key Classes**: `ActInfLearner`
- **Core Concepts**: Active Inference implementation, hierarchical modeling, network-thoughtseed coupling

### [Free Energy Calculations](./free_energy_calculations.md) ⭐
- **Variational Free Energy**: Prediction error minimization (Equation 2)
- **Expected Free Energy**: State transition probabilities (Equation 4)
- **Precision Weighting**: Meta-awareness modulation
- **Complexity Penalties**: Model parsimony enforcement

### [Network Dynamics](./network_dynamics.md)
- **Network Activation Computation**: Four-network model (DMN, VAN, DAN, FPN)
- **Bidirectional Coupling**: Bottom-up and top-down influences
- **Experience-Dependent Modulation**: Expert vs novice differences
- **Anticorrelation Mechanisms**: DMN-DAN competitive dynamics

---

## Foundation Systems

### [Rules-Based Foundation](./rules_based_foundation.md)
- **Primary Module**: `rules_based_learner.py`
- **Base Class**: `RuleBasedLearner`
- **Core Methods**: Target activation generation, meta-awareness calculation

### [Configuration System](./configuration_system.md)
- **Primary Module**: `meditation_config.py`
- **Data Classes**: Comprehensive parameter management
- **Experience Profiles**: Novice vs expert configurations
- **Network Profiles**: Literature-based network expectations

### [State Transitions](./state_transitions.md)
- **Transition Logic**: Natural vs forced transitions
- **Free Energy Minimization**: Equation 4 implementation
- **Threshold Systems**: Experience-dependent criteria
- **Temporal Dynamics**: Dwell times and transition smoothing

---

## Supporting Systems

### [Thoughtseed Dynamics](./thoughtseed_dynamics.md)
- **Agent-Like Entities**: Competitive attention allocation
- **Markov Blankets**: Predictive processing boundaries
- **Network Modulation**: Bidirectional influence patterns
- **Experience Effects**: Expert vs novice thoughtseed behavior

### [Visualization System](./visualization_system.md)
- **Primary Module**: `act_inf_plots.py`
- **Plot Types**: Radar charts, time series, hierarchical displays
- **Data Integration**: JSON-based plotting from saved data
- **Comparative Analysis**: Expert-novice visualizations

### [Utilities and Helpers](./utilities_and_helpers.md)
- **Primary Module**: `meditation_utils.py`
- **Core Functions**: Directory management, JSON serialization
- **Data Conversion**: NumPy to JSON compatibility
- **Output Management**: Structured data saving

---

## Quick Reference

### Key Equations Implemented

1. **Network Activations** (Equation 1):
   ```
   n_k(t) = (1-ζ)∑_i W_{ik}z_i(t) + ζ μ_k(s_t)
   ```

2. **Variational Free Energy** (Equation 2):
   ```
   F_t(s) = ∑_k Π_k(ψ_t)[n_k(t) - μ_k(s_t)]² + λ ||W||_F²
   ```

3. **Weight Updates** (Equation 3):
   ```
   W_ik ← (1-ρ)W_ik + η δ_k(t)z_i(t)
   ```

4. **State Transitions** (Equation 4):
   ```
   P(s_{t+1}=s') ∝ exp(-β F_t(s')) × Θ(s_t → s')
   ```

### Critical Implementation Details

- **Free Energy Calculation**: Located in `ActInfLearner.calculate_free_energy()`
- **Network Computation**: Located in `ActInfLearner.compute_network_activations()`
- **State Transitions**: Implemented in `ActInfLearner.train()` main loop
- **Parameter Learning**: Located in `ActInfLearner.update_network_profiles()`

### Experience Level Differences

| Parameter | Novice | Expert |
|-----------|--------|--------|
| Precision Weight | 0.4 | 0.5 |
| Complexity Penalty | 0.4 | 0.2 |
| Learning Rate | 0.01 | 0.02 |
| Memory Factor | 0.7 | 0.85 |
| DMN Suppression | Moderate | Strong |
| DAN Enhancement | Moderate | Strong |

---

## Navigation Guide

- 🔥 **Start Here**: [Free Energy Calculations](./free_energy_calculations.md) - Core theoretical implementation
- 🧠 **Architecture**: [Active Inference Core](./active_inference_core.md) - Main system overview
- ⚙️ **Configuration**: [Configuration System](./configuration_system.md) - Parameter management
- 📊 **Analysis**: [Visualization System](./visualization_system.md) - Data analysis tools
- 🔄 **Dynamics**: [Network Dynamics](./network_dynamics.md) - Neural network modeling

---

*This documentation corresponds to the theoretical framework presented in the associated research paper on hierarchical computational modeling of meditation expertise.*
