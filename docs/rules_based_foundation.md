# Rules-Based Foundation Documentation

## Overview

This document provides comprehensive technical details on the `RuleBasedLearner` class that serves as the foundation for the Active Inference implementation. This class provides core methods for thoughtseed dynamics, meta-awareness calculation, and state management that are inherited and extended by the `ActInfLearner`.

---

## Core Class: RuleBasedLearner

### Location and Purpose
- **File**: `rules_based_learner.py`
- **Class**: `RuleBasedLearner`
- **Purpose**: Foundation class providing rule-based thoughtseed dynamics and core meditation functionality
- **Lines**: 16-119

### Class Architecture

```python
class RuleBasedLearner:
    """
    Foundation class implementing rule-based thoughtseed dynamics.
    
    This class provides core methods for thoughtseed behavior, meta-awareness,
    and state transitions, serving as a foundation for the active inference implementation.
    It models meditation dynamics using rule-based interactions between thoughtseeds.
    """
```

**Key Design Principles**:
- **Modularity**: Clear separation between foundation and extension
- **Extensibility**: Designed to be inherited and extended
- **Configuration-Driven**: Uses centralized configuration system
- **Experience-Aware**: Supports novice/expert differentiation

---

## Initialization and Setup

### Constructor Method

```python
def __init__(self, experience_level='novice', timesteps_per_cycle=200):
```

#### Core Parameter Initialization

```python
# Core parameters
self.experience_level = experience_level
self.timesteps = timesteps_per_cycle
self.thoughtseeds = THOUGHTSEEDS
self.states = STATES
self.num_thoughtseeds = len(self.thoughtseeds)
```

#### State Tracking Setup

```python
# State tracking
self.state_indices = {state: i for i, state in enumerate(self.states)}
self.transition_counts = defaultdict(lambda: defaultdict(int))
self.natural_transition_count = 0
self.forced_transition_count = 0
```

**Key Tracking Variables**:
- `state_indices`: Numerical mapping for states
- `transition_counts`: Matrix tracking state-to-state transitions
- `natural_transition_count`: Transitions driven by model dynamics
- `forced_transition_count`: Fallback sequential transitions

#### History Management

```python
# History tracking
self.activations_history = []
self.state_history = []
self.meta_awareness_history = []
self.dominant_ts_history = []
self.state_history_over_time = []
```

These lists maintain complete time series data for analysis and visualization.

#### Configuration Integration

```python
# Get noise level from config
aif_params = ActiveInferenceConfig.get_params(experience_level)
self.noise_level = aif_params['noise_level']
```

The foundation class integrates with the centralized configuration system for parameter management.

---

## Core Methods

### 1. Target Activation Generation

#### Method Signature
```python
def get_target_activations(self, state, meta_awareness):
```

#### Purpose and Implementation

This method generates ideal thoughtseed activation patterns for each meditation state, serving as the foundation for both rule-based and active inference dynamics.

```python
def get_target_activations(self, state, meta_awareness):
    """
    Generate target activations for each thoughtseed based on state and meta-awareness.        
    This determines the ideal activation pattern for each state,
    which is then modulated by neural network dynamics.
    """
    # Get target activations from the parameter class
    targets_dict = ThoughtseedParams.get_target_activations(
        state, meta_awareness, self.experience_level)
    
    # Convert dictionary to numpy array in the correct order
    target_activations = np.zeros(self.num_thoughtseeds)
    for i, ts in enumerate(self.thoughtseeds):
        target_activations[i] = targets_dict[ts]
    
    # Add noise for biological plausibility
    target_activations += np.random.normal(0, self.noise_level, size=self.num_thoughtseeds)
    
    # Ensure values are in proper range
    return np.clip(target_activations, 0.05, 1.0)
```

#### Key Features

1. **Configuration Integration**: Uses `ThoughtseedParams.get_target_activations()`
2. **Experience Differentiation**: Applies experience-specific adjustments
3. **Meta-Awareness Modulation**: Activations adjusted by current meta-awareness level
4. **Biological Noise**: Adds realistic variability to prevent fixed patterns
5. **Bound Constraints**: Ensures activations remain in biologically plausible range [0.05, 1.0]

#### Target Activation Patterns

| State | breath_focus | equanimity | pain_discomfort | pending_tasks | self_reflection |
|-------|--------------|------------|-----------------|---------------|------------------|
| **Breath Control** | 0.7 | 0.3 | 0.15 | 0.1 | 0.2 |
| **Mind Wandering** | 0.1 | 0.1 | 0.6 | 0.7 | 0.1 |
| **Meta Awareness** | 0.2 | 0.3 | 0.15 | 0.15 | 0.8 |
| **Redirect Breath** | 0.6 | 0.7 | 0.2 | 0.1 | 0.4 |

**Expert Adjustments**:
- **Breath Control**: +0.1 breath_focus, +0.15 equanimity
- **Mind Wandering**: -0.1 pain_discomfort, -0.25 pending_tasks
- **Meta Awareness**: +0.1 self_reflection

### 2. State Dwell Time Management

#### Method Signature
```python
def get_dwell_time(self, state):
```

#### Implementation Logic

```python
def get_dwell_time(self, state):
    """
    Generate a random dwell time for the given state, based on experience level.
    """
    # Get the configured range from STATE_DWELL_TIMES
    config_min, config_max = STATE_DWELL_TIMES[self.experience_level][state]
    
    # Ensure minimal biological plausibility while respecting configured values
    if state in ['meta_awareness', 'redirect_breath']:
        # For brief states: at least 1 timestep, respect configured max
        min_biological = 1
        max_biological = config_max
    else:
        # For longer states: at least 3 timesteps, respect configured max
        min_biological = 3
        max_biological = config_max
    
    # Generate dwell time with proper constraints
    return max(min_biological, min(max_biological, np.random.randint(config_min, config_max + 1)))
```

#### Dwell Time Patterns

| State | Novice Range | Expert Range | Biological Minimum |
|-------|--------------|--------------|-------------------|
| **Breath Control** | 5-15 | 15-25 | 3 |
| **Mind Wandering** | 15-30 | 8-12 | 3 |
| **Meta Awareness** | 2-5 | 1-3 | 1 |
| **Redirect Breath** | 2-5 | 1-3 | 1 |

**Key Patterns**:
- **Experts**: Longer sustained states, shorter transition states
- **Novices**: Shorter sustained states, longer distracted states
- **Biological Constraints**: Minimum dwell times prevent unrealistic rapid switching

### 3. Meta-Awareness Calculation

#### Method Signature
```python
def get_meta_awareness(self, current_state, activations):
```

#### Implementation Strategy

```python
def get_meta_awareness(self, current_state, activations):
    """
    Calculate meta-awareness based on state and thoughtseed activations.
    
    Meta-awareness is higher in experts, higher during meta_awareness state,
    and influenced by self-reflection and equanimity thoughtseeds.
    """
    # Convert activations array to dictionary for parameter class
    activations_dict = {}
    for i, ts in enumerate(self.thoughtseeds):
        activations_dict[ts] = activations[i]
    
    # Get meta-awareness from parameter class
    meta_awareness = MetacognitionParams.calculate_meta_awareness(
        current_state, activations_dict, self.experience_level)
    
    # Add small random noise for variability
    meta_awareness += np.random.normal(0, 0.05)
    
    return np.clip(meta_awareness, 0.2, 0.85 if self.experience_level == 'novice' else 0.9)
```

#### Meta-Awareness Components

1. **Base State Awareness**:
   - Breath Control: 0.5 (moderate awareness during focus)
   - Mind Wandering: 0.25 (low awareness during distraction)
   - Meta Awareness: 0.7 (high awareness during detection)
   - Redirect Breath: 0.6 (moderate-high during redirection)

2. **Thoughtseed Influences**:
   - Self-reflection: +0.2 × activation
   - Equanimity: +0.15 × activation

3. **Experience Boost**:
   - Novices: +0.0
   - Experts: +0.2

4. **Expert Efficiency**:
   - Meta-awareness state: ×0.8 (more efficient processing)
   - Other states: ×0.9 (better background awareness)

#### Calculation Example

For an expert in meta_awareness state with self_reflection=0.6, equanimity=0.4:

```
base_awareness = 0.7
thoughtseed_boost = 0.6 × 0.2 + 0.4 × 0.15 = 0.18
experience_boost = 0.2
total = 0.7 + 0.18 + 0.2 = 1.08
expert_efficiency = 1.08 × 0.8 = 0.864
final = clip(0.864, 0.2, 0.9) = 0.864
```

---

## Foundation Integration Points

### Configuration System Integration

The `RuleBasedLearner` tightly integrates with the configuration system:

```python
from meditation_config import (
    THOUGHTSEEDS, STATES, STATE_DWELL_TIMES, 
    ActiveInferenceConfig, ThoughtseedParams, MetacognitionParams
)
```

**Key Integration Points**:
- **Parameter Loading**: Uses `ActiveInferenceConfig.get_params()`
- **Target Activations**: Delegates to `ThoughtseedParams.get_target_activations()`
- **Meta-Awareness**: Uses `MetacognitionParams.calculate_meta_awareness()`
- **Dwell Times**: Uses `STATE_DWELL_TIMES` configuration

### Inheritance Architecture

The `RuleBasedLearner` is designed to be extended:

```python
class ActInfLearner(RuleBasedLearner):
    def __init__(self, experience_level='novice', timesteps_per_cycle=200):
        # Initialize base class
        super().__init__(experience_level, timesteps_per_cycle)
        # Add Active Inference specific functionality
```

**Inheritance Benefits**:
- **Code Reuse**: Core functionality shared between implementations
- **Consistent Interface**: Same method signatures across implementations
- **Modular Extension**: Active Inference features added without duplicating foundation code

---

## Experience-Level Differentiation

### Novice Characteristics

```python
if self.experience_level == 'novice':
    # Shorter dwell times in focused states
    # Longer dwell times in distracted states
    # Lower meta-awareness baseline
    # Higher biological noise
    # Less stable thoughtseed patterns
```

**Novice Parameters**:
- `noise_level = 0.06` (higher variability)
- Meta-awareness cap: 0.85
- Shorter focused state durations
- Longer mind-wandering episodes

### Expert Characteristics

```python
if self.experience_level == 'expert':
    # Longer dwell times in focused states
    # Shorter dwell times in distracted states
    # Higher meta-awareness baseline
    # Lower biological noise
    # More stable thoughtseed patterns
```

**Expert Parameters**:
- `noise_level = 0.03` (lower variability)
- Meta-awareness cap: 0.9
- Longer focused state durations
- Shorter mind-wandering episodes
- Enhanced efficiency factors

---

## Data Management and Tracking

### History Management

The foundation class maintains comprehensive tracking:

```python
# Core activation tracking
self.activations_history = []          # Full thoughtseed activation time series
self.state_history = []                # State sequence over time
self.meta_awareness_history = []       # Meta-awareness time series
self.dominant_ts_history = []          # Dominant thoughtseed sequence
self.state_history_over_time = []      # Numerical state sequence
```

### Transition Tracking

```python
# Transition analysis
self.transition_counts = defaultdict(lambda: defaultdict(int))
self.natural_transition_count = 0      # Model-driven transitions
self.forced_transition_count = 0       # Sequential fallback transitions
```

### Additional Tracking Variables

```python
# Pattern analysis
self.transition_activations = {state: [] for state in self.states}
self.distraction_buildup_rates = []    # Distraction growth patterns
```

---

## Validation and Quality Assurance

### Parameter Bounds Checking

```python
# Ensure values are in proper range
return np.clip(target_activations, 0.05, 1.0)

# Meta-awareness bounds
return np.clip(meta_awareness, 0.2, 0.85 if self.experience_level == 'novice' else 0.9)
```

### Biological Plausibility

1. **Activation Bounds**: All thoughtseed activations constrained to [0.05, 1.0]
2. **Meta-Awareness Bounds**: Realistic awareness levels [0.2, 0.9]
3. **Dwell Time Constraints**: Minimum biologically plausible state durations
4. **Noise Incorporation**: Realistic biological variability

### Experience-Level Validation

The foundation class ensures consistent experience-level differentiation:

```python
assert self.experience_level in ['novice', 'expert'], "Invalid experience level"
```

---

## Foundation vs Extension Responsibilities

### Foundation Responsibilities (RuleBasedLearner)

1. **Basic State Management**: State tracking and history maintenance
2. **Target Activation Generation**: Ideal thoughtseed patterns for each state
3. **Meta-Awareness Calculation**: Basic awareness computation
4. **Dwell Time Management**: State duration control
5. **Configuration Integration**: Parameter loading and management
6. **Data Tracking**: History and transition data collection

### Extension Responsibilities (ActInfLearner)

1. **Network Dynamics**: Four-network model implementation
2. **Free Energy Calculation**: Variational and expected free energy
3. **Active Learning**: Network profile updates based on prediction errors
4. **Bidirectional Coupling**: Network-thoughtseed interactions
5. **Advanced State Transitions**: Free energy-driven transition logic
6. **Experience-Specific Modulations**: Enhanced expert-novice differences

---

## Method Call Patterns

### Typical Usage in ActInfLearner

```python
class ActInfLearner(RuleBasedLearner):
    def train(self):
        for t in range(self.timesteps):
            # 1. Use foundation method for meta-awareness
            meta_awareness = self.get_meta_awareness(current_state, activations)
            
            # 2. Use foundation method for target activations  
            target_activations = self.get_target_activations(current_state, meta_awareness)
            
            # 3. Apply Active Inference extensions
            network_acts = self.compute_network_activations(...)
            free_energy = self.calculate_free_energy(...)
            
            # 4. Use foundation method for dwell time management
            if current_dwell >= self.get_dwell_time(current_state):
                # Handle transitions
```

### Override Patterns

The foundation methods can be overridden for specialized behavior:

```python
class AdvancedLearner(RuleBasedLearner):
    def get_meta_awareness(self, current_state, activations):
        # Call parent implementation
        base_awareness = super().get_meta_awareness(current_state, activations)
        
        # Apply specialized modifications
        specialized_awareness = self.apply_advanced_modulation(base_awareness)
        
        return specialized_awareness
```

---

## Performance Considerations

### Computational Efficiency

The foundation class is designed for efficiency:

```python
# Pre-compute indices for fast access
self.state_indices = {state: i for i, state in enumerate(self.states)}

# Use numpy arrays for vectorized operations
target_activations = np.zeros(self.num_thoughtseeds)
```

### Memory Management

```python
# Efficient data structures for history tracking
self.activations_history = []  # List of numpy arrays
self.transition_counts = defaultdict(lambda: defaultdict(int))  # Sparse matrix
```

### Scalability

The foundation class supports scaling:
- **Timesteps**: Handles simulations from 100 to 10,000+ timesteps
- **States**: Extensible to additional meditation states
- **Thoughtseeds**: Supports additional thoughtseed types
- **Experience Levels**: Framework for multiple experience profiles

---

## Testing and Validation

### Unit Test Patterns

```python
def test_target_activations():
    learner = RuleBasedLearner('novice', 200)
    
    # Test breath control activations
    activations = learner.get_target_activations('breath_control', 0.5)
    assert activations[0] > 0.5  # breath_focus should be high
    assert activations[3] < 0.3  # pending_tasks should be low
    
    # Test bounds
    assert np.all(activations >= 0.05)
    assert np.all(activations <= 1.0)
```

### Integration Test Patterns

```python
def test_experience_level_differences():
    novice = RuleBasedLearner('novice', 200)
    expert = RuleBasedLearner('expert', 200)
    
    # Test dwell time differences
    novice_dwell = novice.get_dwell_time('breath_control')
    expert_dwell = expert.get_dwell_time('breath_control')
    
    # Experts should generally have longer focused states
    assert expert_dwell >= novice_dwell or expert_dwell >= 15
```

---

*This documentation covers the foundational `RuleBasedLearner` class that provides the core meditation simulation functionality inherited by the Active Inference implementation. The clean separation between foundation and extension allows for modular development and testing of different theoretical approaches.*
