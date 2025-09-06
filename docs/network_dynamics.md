# Network Dynamics Documentation

## Overview

This document provides comprehensive technical details on the four-network model implementation, covering network activation computations, bidirectional coupling mechanisms, and experience-dependent modulations. The system models the interplay between Default Mode Network (DMN), Ventral Attention Network (VAN), Dorsal Attention Network (DAN), and Frontoparietal Network (FPN).

---

## Four-Network Architecture

### Network Definitions

| Network | Abbreviation | Primary Function | Literature Basis |
|---------|--------------|------------------|------------------|
| **Default Mode Network** | DMN | Self-referential processing, mind-wandering | Raichle et al., 2001; Buckner et al., 2008 |
| **Ventral Attention Network** | VAN | Salience detection, bottom-up attention | Corbetta & Shulman, 2002; Fox et al., 2006 |
| **Dorsal Attention Network** | DAN | Goal-directed attention, top-down control | Corbetta & Shulman, 2002; Vincent et al., 2008 |
| **Frontoparietal Network** | FPN | Cognitive control, executive function | Vincent et al., 2008; Power et al., 2011 |

### Network Initialization

```python
self.networks = ['DMN', 'VAN', 'DAN', 'FPN']
network_acts = {net: 0.2 for net in self.networks}  # Baseline activation
```

The baseline activation of 0.2 prevents numerical instabilities and represents minimal background activity in each network.

---

## Core Network Computation

### Primary Method: compute_network_activations()

**Location**: `act_inf_learner.py`, lines 79-330

**Theoretical Foundation**: Implements Equation 1 from the framework:
```
n_k(t) = (1-ζ)∑_i W_{ik}z_i(t) + ζ μ_k(s_t)
```

### Implementation Architecture

#### 1. Bidirectional Influence Computation

```python
def compute_network_activations(self, thoughtseed_activations, current_state, meta_awareness):
    # Get thoughtseed-to-network contribution matrix
    ts_to_network = self.learned_network_profiles["thoughtseed_contributions"]
    
    # Initialize with baseline activations
    network_acts = {net: 0.2 for net in self.networks}
    
    # Experience-dependent weighting
    if self.experience_level == 'expert':
        bottom_up_weight = 0.5    # ζ in equation
        top_down_weight = 0.5     # (1-ζ) in equation
    else:
        bottom_up_weight = 0.6    # More stimulus-driven
        top_down_weight = 0.4     # Less cognitive control
```

#### 2. Bottom-Up Processing (Thoughtseeds → Networks)

```python
# Calculate bottom-up influence: thoughtseeds -> networks
for i, ts in enumerate(self.thoughtseeds):
    ts_act = thoughtseed_activations[i]
    for net in self.networks:
        # Weighted contribution based on thoughtseed activation
        network_acts[net] += ts_act * ts_to_network[ts][net] * bottom_up_weight
```

**Key Features**:
- Each thoughtseed contributes to networks based on learned weights
- Bottom-up weight is higher in novices (0.6 vs 0.5)
- Reflects stimulus-driven attention in less experienced practitioners

#### 3. Top-Down Processing (State Expectations → Networks)

```python
# Calculate top-down influence: state expectations -> networks
state_expect = self.learned_network_profiles["state_network_expectations"][current_state]
for net in self.networks:
    # Meta-awareness amplifies top-down control
    meta_factor = meta_awareness * (1.2 if self.experience_level == 'expert' else 1.0)
    state_influence = state_expect[net] * meta_factor * top_down_weight
    network_acts[net] = (1 - top_down_weight) * network_acts[net] + state_influence
```

**Key Features**:
- State expectations provide top-down bias
- Meta-awareness amplifies control (more in experts)
- Top-down weight is higher in experts (0.5 vs 0.4)

---

## State-Specific Network Modulations

### Meta-Awareness State

#### VAN Enhancement (Salience Detection)
```python
if current_state == "meta_awareness":
    van_boost_effect = self.network_modulation.get('van_boost', {}).get('meta_awareness')
    if van_boost_effect:
        van_boost = van_boost_effect.get_value(current_state, self.experience_level)
        if self.experience_level == 'expert':
            van_boost *= 1.5  # 50% stronger VAN activation for experts
        network_acts['VAN'] += van_boost * meta_awareness
```

#### FPN Enhancement (Cognitive Control)
```python
    fpn_boost_effect = self.network_modulation.get('fpn_boost', {}).get('meta_awareness')
    if fpn_boost_effect:
        fpn_boost = fpn_boost_effect.get_value(current_state, self.experience_level)
        network_acts['FPN'] += fpn_boost * meta_awareness
```

#### DMN Suppression (Default Mode Reduction)
```python
    dmn_suppression_effect = self.network_modulation.get('dmn_suppression', {}).get('meta_awareness')
    if dmn_suppression_effect:
        dmn_suppress = dmn_suppression_effect.get_value(current_state, self.experience_level)
        if self.experience_level == 'expert':
            dmn_suppress *= 1.5  # 50% stronger suppression for experts
        network_acts['DMN'] *= (1.0 - dmn_suppress)
```

#### Expert-Specific DMN Adjustment
```python
    if self.experience_level == 'expert':
        expected_dmn = self.learned_network_profiles["state_network_expectations"][current_state]["DMN"]
        current_dmn = network_acts['DMN']
        target_dmn = 0.2  # Literature-based target
        
        # Weighted averaging (30% current, 70% target)
        network_acts['DMN'] = 0.3 * current_dmn + 0.7 * target_dmn
        
        # Cap at maximum allowed DMN for experts
        max_dmn = 0.25
        network_acts['DMN'] = min(network_acts['DMN'], max_dmn)
```

### Mind Wandering State

#### DMN Boost (Default Mode Enhancement)
```python
elif current_state == "mind_wandering":
    dmn_boost_effect = self.network_modulation.get('dmn_boost', {}).get('mind_wandering')
    if dmn_boost_effect:
        dmn_boost = dmn_boost_effect.get_value(current_state, self.experience_level)
        if self.experience_level == 'novice':
            dmn_boost *= 1.3  # 30% stronger for novices
        network_acts['DMN'] += dmn_boost
```

#### DAN Suppression (Attention Reduction)
```python
    dan_suppression = self.network_modulation.get('dan_suppression', {}).get('mind_wandering')
    if dan_suppression:
        dan_suppress_value = dan_suppression.get_value(current_state, self.experience_level)
        if self.experience_level == 'novice':
            dan_suppress_value *= 1.25  # 25% stronger for novices
        network_acts['DAN'] *= (1.0 - dan_suppress_value)
```

### Focused States (Breath Control, Redirect Breath)

#### DAN Boost (Attention Enhancement)
```python
elif current_state in ["breath_control", "redirect_breath"]:
    dan_boost_effect = self.network_modulation.get('dan_boost', {}).get(current_state)
    if dan_boost_effect:
        dan_boost = dan_boost_effect.get_value(current_state, self.experience_level)
        if self.experience_level == 'expert':
            dan_boost *= 1.3  # 30% stronger for experts
        network_acts['DAN'] += dan_boost * meta_awareness
```

#### DMN Suppression (Default Mode Reduction)
```python
    dmn_suppression_effect = self.network_modulation.get('dmn_suppression', {}).get(current_state)
    if dmn_suppression_effect:
        dmn_suppress = dmn_suppression_effect.get_value(current_state, self.experience_level)
        if self.experience_level == 'expert':
            dmn_suppress *= 1.5  # 50% stronger for experts
        network_acts['DMN'] *= (1.0 - dmn_suppress)
```

---

## Expert-Specific Network Enhancements

### FPN-Mediated Top-Down Control

```python
if self.experience_level == 'expert':
    # Experts have stronger top-down control from FPN
    fpn_influence = self.fpn_enhancement * 0.2  # fpn_enhancement = 1.2 for experts
    for net in ['DMN', 'VAN', 'DAN']:
        network_acts[net] = ((1.0 - fpn_influence) * network_acts[net] + 
                            fpn_influence * network_acts['FPN'])
```

This mechanism reflects enhanced executive control in expert meditators, where the FPN exerts regulatory influence over other networks.

### Enhanced Network Separation

Experts demonstrate stronger network differentiation through enhanced modulation factors:

| Modulation Type | Novice Factor | Expert Factor |
|----------------|---------------|---------------|
| DMN Suppression (Meta-Awareness) | 1.0x | 1.5x |
| DAN Boost (Focused States) | 1.0x | 1.3x |
| VAN Boost (Meta-Awareness) | 1.0x | 1.5x |
| Anticorrelation Strength | 1.0x | 1.5x |

---

## DMN-DAN Anticorrelation Mechanism

### Theoretical Background

The DMN-DAN anticorrelation is a well-established phenomenon in neuroscience literature (Fox et al., 2005; Spreng et al., 2013). Our implementation models this competitive relationship.

### Implementation

```python
# Get anticorrelation strengths
dan_dmn_anticorr_strength = 0.18  # Default
dmn_dan_anticorr_strength = 0.15  # Default

if self.experience_level == 'expert':
    dan_dmn_anticorr_strength *= 1.5  # 50% stronger
    dmn_dan_anticorr_strength *= 1.5  # 50% stronger

# DMN anticorrelation with asymptotic approach
anticorr_effect = dan_dmn_anticorr_strength * (network_acts['DAN'] - 0.5)
if anticorr_effect > 0 and network_acts['DMN'] < boundary_threshold:
    anticorr_effect *= (network_acts['DMN'] / boundary_threshold)
elif anticorr_effect < 0 and network_acts['DMN'] > (1.0 - boundary_threshold):
    anticorr_effect *= (1.0 - (network_acts['DMN'] - (1.0 - boundary_threshold)) / boundary_threshold)
network_acts['DMN'] = max(0.05, min(0.95, network_acts['DMN'] - anticorr_effect))
```

### Key Features

1. **Asymptotic Boundary Behavior**: Effects diminish near activation boundaries
2. **Experience-Dependent Strength**: Experts show stronger anticorrelation
3. **Bidirectional Influence**: Both DMN → DAN and DAN → DMN effects
4. **Biological Bounds**: Activations constrained to [0.05, 0.95] range

---

## Temporal Dynamics and Memory

### Memory Factor Application

```python
if hasattr(self, 'prev_network_acts') and self.prev_network_acts:
    for net in self.networks:
        network_acts[net] = (self.memory_factor * self.prev_network_acts[net] + 
                            (1 - self.memory_factor) * network_acts[net])
```

**Memory Parameters**:
- **Novices**: `memory_factor = 0.7` (less temporal stability)
- **Experts**: `memory_factor = 0.85` (more temporal stability)

### Temporal Smoothing for Volatile Networks

```python
if len(self.network_activations_history) > 3:
    volatility_nets = ['DMN', 'DAN']
    dmn_dan_smoothing = 0.25 if self.experience_level == 'novice' else 0.35
    
    for net in volatility_nets:
        recent_values = [self.network_activations_history[-i][net] 
                        for i in range(1, min(4, len(self.network_activations_history)+1))]
        network_acts[net] = ((1-dmn_dan_smoothing) * network_acts[net] + 
                            dmn_dan_smoothing * np.mean(recent_values))
```

This mechanism prevents excessive volatility in DMN and DAN activations while maintaining responsiveness to changes.

---

## Non-Linear Dynamics

### Compression Functions

The implementation includes non-linear compression to prevent extreme activations:

```python
# Get compression parameters
high_threshold = self.non_linear_dynamics.high_compression_threshold  # 0.8
low_threshold = self.non_linear_dynamics.low_compression_threshold    # 0.2
high_factor = self.non_linear_dynamics.high_compression_factor        # 0.2
low_factor = self.non_linear_dynamics.low_compression_factor          # 0.2

for net in ['DMN', 'DAN']:
    if network_acts[net] > high_threshold:
        # Compress high values
        compression = (network_acts[net] - high_threshold) * high_factor
        network_acts[net] = high_threshold + compression
    elif network_acts[net] < low_threshold:
        # Compress low values
        compression = (low_threshold - network_acts[net]) * low_factor
        network_acts[net] = low_threshold - compression
```

### Rapid Change Detection

For expert meditators during mind-wandering, the system applies additional smoothing when rapid changes are detected:

```python
if current_state == "mind_wandering" and self.experience_level == 'expert':
    if len(self.network_activations_history) > 3:
        rapid_change_threshold = 0.15
        rapid_change_smoothing = 0.7
        
        for net in ['DMN', 'DAN']:
            recent = self.network_activations_history[-1][net]
            current = network_acts[net]
            if abs(current - recent) > rapid_change_threshold:
                network_acts[net] = (rapid_change_smoothing * recent + 
                                   (1.0 - rapid_change_smoothing) * current)
```

---

## Biological Plausibility Constraints

### Activation Bounds

```python
# Normalize and add noise
for net in self.networks:
    network_acts[net] = np.clip(network_acts[net], 0.05, 1.0)
    # Add small noise for biological plausibility
    network_acts[net] += np.random.normal(0, self.noise_level)
    network_acts[net] = np.clip(network_acts[net], 0.05, 1.0)
```

### VAN Physiological Limits

```python
# VAN values > 0.85 are neurophysiologically implausible
max_van = 0.85
if network_acts['VAN'] > max_van:
    network_acts['VAN'] = max_van
```

### Network Relationship Constraints

```python
if current_state == "meta_awareness" and self.experience_level == 'expert':
    # Ensure FPN doesn't exceed VAN (literature shows VAN activation precedes FPN)
    if network_acts['FPN'] > network_acts['VAN']:
        network_acts['FPN'] = network_acts['VAN'] * 0.95
    
    # Ensure DMN-DAN anticorrelation is maintained
    if network_acts['DMN'] > 0.3 and network_acts['DAN'] > 0.5:
        network_acts['DMN'] *= 0.85
```

---

## Network Profile Learning

### Thoughtseed-to-Network Mapping

The system learns how thoughtseeds contribute to network activations:

```python
self.learned_network_profiles = {
    "thoughtseed_contributions": {ts: {} for ts in self.thoughtseeds},
    "state_network_expectations": {state: {} for state in self.states}
}
```

### Initial Network Profiles

Based on meditation neuroscience literature:

```python
NETWORK_PROFILES = {
    "thoughtseed_contributions": {
        "breath_focus": NetworkProfile(DMN=0.2, VAN=0.3, DAN=0.8, FPN=0.6),
        "pain_discomfort": NetworkProfile(DMN=0.5, VAN=0.7, DAN=0.3, FPN=0.4),
        "pending_tasks": NetworkProfile(DMN=0.8, VAN=0.5, DAN=0.2, FPN=0.4),
        "self_reflection": NetworkProfile(DMN=0.6, VAN=0.4, DAN=0.3, FPN=0.8),
        "equanimity": NetworkProfile(DMN=0.3, VAN=0.3, DAN=0.5, FPN=0.9)
    }
}
```

### Profile Updates

Network profiles are updated based on prediction errors:

```python
def update_network_profiles(self, thoughtseed_activations, network_activations, 
                           current_state, prediction_errors):
    for i, ts in enumerate(self.thoughtseeds):
        if thoughtseed_activations[i] > 0.2:  # Only update when active
            for net in self.networks:
                error_sign = 1 if network_activations[net] < expected[net] else -1
                update = self.learning_rate * (error_sign * current_error) * ts_act / precision
                self.learned_network_profiles["thoughtseed_contributions"][ts][net] += update
```

---

## State-Dependent Network Expectations

### Experience-Level Profiles

The system maintains separate network expectation profiles for novices and experts:

```python
"state_expected_profiles": {
    "breath_control": {
        "novice": NetworkProfile(DMN=0.35, VAN=0.4, DAN=0.7, FPN=0.5),
        "expert": NetworkProfile(DMN=0.2, VAN=0.4, DAN=0.85, FPN=0.7)
    },
    "mind_wandering": {
        "novice": NetworkProfile(DMN=0.85, VAN=0.45, DAN=0.2, FPN=0.35),
        "expert": NetworkProfile(DMN=0.65, VAN=0.5, DAN=0.35, FPN=0.55)
    }
    # ... additional states
}
```

### Key Patterns

1. **DMN Suppression**: Experts show lower DMN across all states
2. **DAN Enhancement**: Experts show higher DAN in focused states
3. **FPN Control**: Experts show higher FPN across states
4. **VAN Sensitivity**: Similar VAN levels with slight expert enhancement

---

## Integration with Thoughtseed Modulation

### Bidirectional Coupling

The network dynamics feed back to modulate thoughtseed activations:

```python
def network_modulated_activations(self, activations, network_acts, current_state):
    # DMN enhances pending_tasks and self_reflection
    dmn_strength = network_acts['DMN']
    modulated_acts[pending_tasks_idx] += dmn_pending_value * dmn_strength
    
    # DAN enhances breath_focus, suppresses distractions
    dan_strength = network_acts['DAN']
    modulated_acts[breath_focus_idx] += dan_breath_value * dan_strength
    modulated_acts[pending_tasks_idx] -= dan_pending_value * dan_strength
    
    # VAN enhances pain_discomfort (salience)
    van_strength = network_acts['VAN']
    modulated_acts[pain_discomfort_idx] += van_pain_value * van_strength
    
    # FPN enhances self_reflection and equanimity
    fpn_strength = network_acts['FPN']
    modulated_acts[self_reflection_idx] += fpn_reflection_value * fpn_strength
    modulated_acts[equanimity_idx] += fpn_equanimity_value * fpn_strength
```

This creates a closed-loop system where networks and thoughtseeds mutually influence each other.

---

## Validation and Debugging

### Network Value Monitoring

```python
print(f"\n{self.experience_level.upper()} NETWORK VALUES BY STATE:")
for state in self.states:
    print(f"  {state}:")
    state_networks = {
        net: float(np.mean([
            self.network_activations_history[j][net]
            for j, s in enumerate(self.state_history) if s == state
        ])) for net in self.networks
    }
    for net in self.networks:
        print(f"    {net}: {state_networks[net]:.2f}")
```

### Expected Value Ranges

| Network | State | Novice Range | Expert Range |
|---------|-------|--------------|--------------|
| DMN | Breath Control | 0.30-0.40 | 0.15-0.25 |
| DMN | Mind Wandering | 0.75-0.90 | 0.55-0.70 |
| DAN | Breath Control | 0.65-0.75 | 0.80-0.90 |
| DAN | Mind Wandering | 0.15-0.25 | 0.30-0.40 |
| VAN | Meta Awareness | 0.65-0.75 | 0.75-0.85 |
| FPN | All States | 0.40-0.60 | 0.55-0.75 |

### Quality Assurance Checks

```python
# Ensure reasonable network relationships
assert network_acts['DMN'] + network_acts['DAN'] < 1.5, "DMN+DAN too high"
assert abs(network_acts['VAN'] - network_acts['FPN']) < 0.4, "VAN-FPN divergence"
```

---

## Performance Considerations

### Computational Efficiency

The network computation is optimized through:

1. **Vectorized Operations**: Using NumPy arrays where possible
2. **Conditional Updates**: Only applying modulations when necessary
3. **Bounded Calculations**: Preventing expensive overflow/underflow operations

### Memory Management

Network activations are stored efficiently:

```python
self.network_activations_history.append(network_acts.copy())  # Explicit copy
```

### Parallelization Potential

Future optimizations could parallelize:
- State-specific modulation calculations
- Multiple network update computations
- Experience-level comparative simulations

---

*This documentation covers the comprehensive network dynamics implementation that forms the neural substrate of the meditation simulation framework. The four-network model provides a biologically plausible foundation for modeling attentional states and their modulation by contemplative expertise.*
