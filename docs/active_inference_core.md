# Active Inference Core Implementation

## Overview

This document provides comprehensive technical documentation for the core Active Inference implementation in the meditation simulation framework. The `ActInfLearner` class extends the rule-based foundation to implement a full Active Inference model of meditation expertise.

---

## Main Class: ActInfLearner

### Location and Inheritance
- **File**: `act_inf_learner.py`
- **Class**: `ActInfLearner`
- **Inherits from**: `RuleBasedLearner`
- **Lines**: 15-977

### Class Architecture

```python
class ActInfLearner(RuleBasedLearner):
    """
    Active Inference extension of the RuleBasedLearner, implementing the three-level
    Thoughtseeds Framework from the paper:
    
    Level 1: Attentional Network Superordinate Ensembles (DMN, VAN, DAN, FPN)
    Level 2: Thoughtseed Dynamics (breath_focus, pending_tasks, etc.)
    Level 3: Metacognitive Regulation (meta-awareness, precision weighting)
    """
```

### Key Theoretical Framework

The implementation models the **four-stage Vipassana cycle**:

1. **Breath Control**: Sustained attention
2. **Mind Wandering**: Attentional drift  
3. **Meta-Awareness**: Noticing the lapse
4. **Redirect Attention**: Returning to breath

This maps to the **Active Inference equations (1-4)** from the theoretical framework.

---

## Initialization and Configuration

### Constructor Method

```python
def __init__(self, experience_level='novice', timesteps_per_cycle=200):
```

#### Initialization Sequence

1. **Base Class Initialization**
   ```python
   super().__init__(experience_level, timesteps_per_cycle)
   ```

2. **Network Configuration**
   ```python
   self.networks = ['DMN', 'VAN', 'DAN', 'FPN']
   self.network_activations_history = []
   self.free_energy_history = []
   self.prediction_error_history = []
   self.precision_history = []
   ```

3. **Parameter Loading**
   ```python
   aif_params = ActiveInferenceConfig.get_params(experience_level)
   self.precision_weight = aif_params['precision_weight']
   self.complexity_penalty = aif_params['complexity_penalty']
   # ... additional parameters
   ```

4. **Profile Initialization**
   ```python
   self.learned_network_profiles = {
       "thoughtseed_contributions": {ts: {} for ts in self.thoughtseeds},
       "state_network_expectations": {state: {} for state in self.states}
   }
   ```

### Experience-Level Parameters

| Parameter | Novice Value | Expert Value | Description |
|-----------|--------------|--------------|-------------|
| `precision_weight` | 0.4 | 0.5 | Meta-awareness influence on precision |
| `complexity_penalty` | 0.4 | 0.2 | Parsimony constraint strength |
| `learning_rate` | 0.01 | 0.02 | Weight adaptation speed |
| `noise_level` | 0.06 | 0.03 | Biological variability |
| `memory_factor` | 0.7 | 0.85 | Temporal smoothing strength |
| `fpn_enhancement` | 1.0 | 1.2 | Executive network boost |

---

## Core Methods

### 1. Network Activation Computation

#### Method Signature
```python
def compute_network_activations(self, thoughtseed_activations, current_state, meta_awareness):
```

#### Theoretical Implementation
Implements **Equation 1** from the framework:
```
n_k(t) = (1-ζ)∑_i W_{ik}z_i(t) + ζ μ_k(s_t)
```

#### Implementation Logic

1. **Initialize Baseline Activations**
   ```python
   network_acts = {net: 0.2 for net in self.networks}  # Prevent zeros
   ```

2. **Experience-Dependent Weighting**
   ```python
   if self.experience_level == 'expert':
       bottom_up_weight = 0.5    # ζ in equation
       top_down_weight = 0.5     # (1-ζ) in equation
   else:
       bottom_up_weight = 0.6    # More stimulus-driven
       top_down_weight = 0.4     # Less top-down control
   ```

3. **Bottom-Up Influence**: Thoughtseeds → Networks
   ```python
   for i, ts in enumerate(self.thoughtseeds):
       ts_act = thoughtseed_activations[i]
       for net in self.networks:
           network_acts[net] += ts_act * ts_to_network[ts][net] * bottom_up_weight
   ```

4. **Top-Down Influence**: State expectations → Networks
   ```python
   state_expect = self.learned_network_profiles["state_network_expectations"][current_state]
   for net in self.networks:
       meta_factor = meta_awareness * (1.2 if self.experience_level == 'expert' else 1.0)
       state_influence = state_expect[net] * meta_factor * top_down_weight
       network_acts[net] = (1 - top_down_weight) * network_acts[net] + state_influence
   ```

#### State-Specific Modulations

**Meta-Awareness State**:
```python
if current_state == "meta_awareness":
    # VAN boost (salience detection)
    network_acts['VAN'] += van_boost * meta_awareness
    
    # FPN boost (cognitive control)
    network_acts['FPN'] += fpn_boost * meta_awareness
    
    # DMN suppression (default mode reduction)
    network_acts['DMN'] *= (1.0 - dmn_suppress)
```

**Mind Wandering State**:
```python
elif current_state == "mind_wandering":
    # DMN enhancement (default mode activation)
    network_acts['DMN'] += dmn_boost
    
    # DAN suppression (attention network reduction)
    network_acts['DAN'] *= (1.0 - dan_suppress_value)
```

**Focused States (Breath Control, Redirect Breath)**:
```python
elif current_state in ["breath_control", "redirect_breath"]:
    # DAN boost (attention network enhancement)
    network_acts['DAN'] += dan_boost * meta_awareness
    
    # DMN suppression (default mode reduction)
    network_acts['DMN'] *= (1.0 - dmn_suppress)
```

#### DMN-DAN Anticorrelation

The implementation enforces the well-established DMN-DAN anticorrelation:

```python
# DMN anticorrelation with asymptotic approach
anticorr_effect = dan_dmn_anticorr_strength * (network_acts['DAN'] - 0.5)
network_acts['DMN'] = max(0.05, min(0.95, network_acts['DMN'] - anticorr_effect))

# DAN anticorrelation 
anticorr_effect = dmn_dan_anticorr_strength * (network_acts['DMN'] - 0.5)
network_acts['DAN'] = max(0.05, min(0.95, network_acts['DAN'] - anticorr_effect))
```

### 2. Free Energy Calculation

**See detailed documentation**: [Free Energy Calculations](./free_energy_calculations.md)

Key method:
```python
def calculate_free_energy(self, network_acts, current_state, meta_awareness):
```

### 3. Network Profile Learning

#### Method Signature
```python
def update_network_profiles(self, thoughtseed_activations, network_activations, 
                           current_state, prediction_errors):
```

#### Theoretical Implementation
Implements **Equation 3**:
```
W_ik ← (1-ρ)W_ik + η δ_k(t)z_i(t)
```

#### Learning Logic

1. **Active Thoughtseed Filter**
   ```python
   if ts_act > 0.2:  # Only update when thoughtseed is significantly active
   ```

2. **Precision Calculation**
   ```python
   precision = 1.0 + (5.0 if self.experience_level == 'expert' else 2.0) * len(self.network_activations_history)/self.timesteps
   ```

3. **Bayesian Update**
   ```python
   error_sign = 1 if network_activations[net] < expected[net] else -1
   update = self.learning_rate * (error_sign * current_error) * ts_act / precision
   self.learned_network_profiles["thoughtseed_contributions"][ts][net] += update
   ```

4. **Weight Bounds**
   ```python
   self.learned_network_profiles["thoughtseed_contributions"][ts][net] = np.clip(
       self.learned_network_profiles["thoughtseed_contributions"][ts][net], 0.1, 0.9)
   ```

### 4. Network-Based Thoughtseed Modulation

#### Method Signature
```python
def network_modulated_activations(self, activations, network_acts, current_state):
```

#### Modulation Rules

**DMN Effects**:
```python
# DMN enhances pending_tasks and self_reflection, suppresses breath_focus
dmn_strength = network_acts['DMN']
modulated_acts[self.thoughtseeds.index('pending_tasks')] += dmn_pending_value * dmn_strength
modulated_acts[self.thoughtseeds.index('self_reflection')] += dmn_reflection_value * dmn_strength
modulated_acts[self.thoughtseeds.index('breath_focus')] -= dmn_breath_value * dmn_strength
```

**VAN Effects**:
```python
# VAN enhances pain_discomfort (salience) and self_reflection during meta_awareness
van_strength = network_acts['VAN']
modulated_acts[self.thoughtseeds.index('pain_discomfort')] += van_pain_value * van_strength
if current_state == "meta_awareness":
    modulated_acts[self.thoughtseeds.index('self_reflection')] += van_reflection_value * van_strength
```

**DAN Effects**:
```python
# DAN enhances breath_focus, suppresses distractions
dan_strength = network_acts['DAN']
modulated_acts[self.thoughtseeds.index('breath_focus')] += dan_breath_value * dan_strength
modulated_acts[self.thoughtseeds.index('pending_tasks')] -= dan_pending_value * dan_strength
modulated_acts[self.thoughtseeds.index('pain_discomfort')] -= dan_pain_value * dan_strength
```

**FPN Effects**:
```python
# FPN enhances self_reflection and equanimity (metacognition and regulation)
fpn_strength = network_acts['FPN']
modulated_acts[self.thoughtseeds.index('self_reflection')] += fpn_reflection_value * fpn_strength
modulated_acts[self.thoughtseeds.index('equanimity')] += fpn_equanimity_value * fpn_strength
```

---

## Main Training Loop

### Method Signature
```python
def train(self):
```

### Training Sequence

1. **Initialization**
   ```python
   # Create directories for output
   ensure_directories()
   
   # Initialize training sequence
   state_sequence = ["breath_control", "mind_wandering", "meta_awareness", "redirect_breath"]
   current_state_index = 0
   current_state = state_sequence[current_state_index]
   ```

2. **Main Simulation Loop**
   ```python
   for t in range(self.timesteps):
       # 1. Calculate meta-awareness
       meta_awareness = self.get_meta_awareness(current_state, activations)
       
       # 2. Get target activations
       target_activations = self.get_target_activations(current_state, meta_awareness)
       
       # 3. Smooth activation transitions
       activations = target_activations * 0.9 + prev_activations * 0.1
       
       # 4. Apply state-specific adjustments
       # ... [state-specific logic]
       
       # 5. Compute network activations
       network_acts = self.compute_network_activations(activations, current_state, meta_awareness)
       
       # 6. Calculate free energy
       free_energy, prediction_errors, total_prediction_error = self.calculate_free_energy(
           network_acts, current_state, meta_awareness)
       
       # 7. Update network profiles
       self.update_network_profiles(activations, network_acts, current_state, prediction_errors)
       
       # 8. Apply network-based modulation
       activations = self.network_modulated_activations(activations, network_acts, current_state)
       
       # 9. Record histories
       self.network_activations_history.append(network_acts.copy())
       self.free_energy_history.append(free_energy)
       # ... [other recordings]
       
       # 10. Handle state transitions
       if current_dwell >= dwell_limit:
           # ... [transition logic - see State Transitions documentation]
   ```

### State Transition Logic

The training loop implements **Equation 4** concepts through natural transition detection:

```python
# Higher free energy increases transition probability
precision_factor = 1.5 if self.experience_level == 'expert' else 0.8
fe_factor = min(0.3, free_energy * 0.3 * precision_factor)
natural_prob = min(0.95, natural_prob + fe_factor)

if np.random.random() < natural_prob:
    # Check state-specific transition conditions
    # ... [detailed in State Transitions documentation]
```

---

## Experience-Level Differences

### Expert Characteristics

1. **Enhanced Top-Down Control**
   ```python
   if self.experience_level == 'expert':
       fpn_influence = self.fpn_enhancement * 0.2
       for net in ['DMN', 'VAN', 'DAN']:
           network_acts[net] = (1.0 - fpn_influence) * network_acts[net] + fpn_influence * network_acts['FPN']
   ```

2. **Stronger DMN Suppression**
   ```python
   if self.experience_level == 'expert':
       current_dmn = network_acts['DMN']
       target_dmn = 0.2  # Literature-based target
       network_acts['DMN'] = 0.3 * current_dmn + 0.7 * target_dmn
   ```

3. **Enhanced Anticorrelation**
   ```python
   if self.experience_level == 'expert':
       dan_dmn_anticorr_strength *= 1.5  # 50% stronger
       dmn_dan_anticorr_strength *= 1.5  # 50% stronger
   ```

### Novice Characteristics

1. **More Bottom-Up Processing**
   ```python
   if self.experience_level == 'novice':
       bottom_up_weight = 0.6  # More stimulus-driven
       top_down_weight = 0.4   # Less cognitive control
   ```

2. **Enhanced Mind-Wandering Patterns**
   ```python
   if current_state == "mind_wandering" and self.experience_level == 'novice':
       pt_idx = self.thoughtseeds.index("pending_tasks")
       activations[pt_idx] *= 1.15  # More dominant pending tasks
   ```

3. **Higher Distraction Susceptibility**
   ```python
   distraction_scale = 2.5 if self.experience_level == 'novice' else 1.2
   distraction_growth = 0.035 * dwell_factor * distraction_scale
   ```

---

## Data Output and Analysis

### History Tracking

The implementation tracks comprehensive data for analysis:

```python
self.network_activations_history.append(network_acts.copy())
self.free_energy_history.append(free_energy)
self.prediction_error_history.append(total_prediction_error)
self.precision_history.append(0.5 + self.precision_weight * meta_awareness)
```

### JSON Output Generation

The training method calls utility functions to save results:

```python
_save_json_outputs(self)  # Detailed data for analysis
```

Output includes:
- Thoughtseed activation time series
- Network activation time series  
- Free energy trajectories
- Transition statistics
- Learned network profiles

### Statistical Analysis

State-specific averages are computed:

```python
'average_network_activations_by_state': {
    state: {
        net: float(np.mean([
            self.network_activations_history[j][net]
            for j, s in enumerate(self.state_history) if s == state
        ])) for net in self.networks
    } for state in self.states if any(s == state for s in self.state_history)
}
```

---

## Integration Points

### With Configuration System

The `ActInfLearner` heavily integrates with `meditation_config.py`:

- **Parameter Loading**: `ActiveInferenceConfig.get_params()`
- **Network Profiles**: `NETWORK_PROFILES`  
- **Transition Thresholds**: Experience-specific values
- **Modulation Effects**: `NetworkModulationConfig`

### With Rules-Based Foundation

Inherits and extends `RuleBasedLearner` methods:

- **Target Activations**: `get_target_activations()`
- **Meta-Awareness**: `get_meta_awareness()`
- **Dwell Times**: `get_dwell_time()`
- **Base Tracking**: History management

### With Visualization System

Generates data consumed by `act_inf_plots.py`:

- **JSON Outputs**: Structured data files
- **Time Series**: Network and thoughtseed trajectories  
- **Statistical Summaries**: State-specific averages
- **Transition Data**: Natural vs forced transitions

---

## Validation and Quality Assurance

### Biological Plausibility Checks

1. **Network Activation Bounds**
   ```python
   network_acts[net] = np.clip(network_acts[net], 0.05, 1.0)
   ```

2. **VAN Implausibility Limits**
   ```python
   max_van = 0.85  # Neurophysiologically reasonable
   if network_acts['VAN'] > max_van:
       network_acts['VAN'] = max_van
   ```

3. **Natural Transition Requirements**
   ```python
   if self.natural_transition_count < 4:
       print(f"WARNING: Only {self.natural_transition_count} natural transitions occurred.")
   ```

### Debug Output

The implementation provides comprehensive debugging information:

```python
print(f"\n{self.experience_level.upper()} NETWORK VALUES BY STATE:")
for state in self.states:
    print(f"  {state}:")
    for net in self.networks:
        print(f"    {net}: {state_networks[net]:.2f}")
```

---

## Performance Considerations

### Computational Complexity

- **Time Complexity**: O(T × S × N × M) where:
  - T = timesteps (200)
  - S = states (4) 
  - N = networks (4)
  - M = thoughtseeds (5)

- **Space Complexity**: O(T × (S + N + M)) for history storage

### Memory Management

The implementation uses efficient numpy arrays and periodic data saving to manage memory usage during long simulations.

### Parallelization Potential

Future optimizations could parallelize:
- Network activation computations
- Free energy calculations across states
- Multiple experience level simulations

---

*This documentation covers the core Active Inference implementation that drives the meditation simulation framework. The methods work together to create a biologically plausible model of expert-novice differences in contemplative practice.*
