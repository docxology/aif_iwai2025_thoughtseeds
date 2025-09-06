# State Transitions Documentation

## Overview

This document provides comprehensive technical details on the state transition mechanics that govern movement between the four meditation states in the Active Inference framework. The system implements free energy minimization principles to drive natural, biologically plausible transitions between breath control, mind wandering, meta-awareness, and attention redirection states.

---

## Four-State Meditation Model

### State Definitions

| State | Code | Duration | Function | Experience Difference |
|-------|------|----------|----------|----------------------|
| **Breath Control** | `breath_control` | Sustained | Focused attention on breath | Experts: longer, more stable |
| **Mind Wandering** | `mind_wandering` | Variable | Attentional drift, distraction | Experts: shorter, less severe |
| **Meta-Awareness** | `meta_awareness` | Brief | Noticing attentional lapse | Experts: more efficient |
| **Redirect Breath** | `redirect_breath` | Brief | Returning to breath focus | Experts: more skillful |

### Vipassana Cycle Implementation

The system models the **four-stage Vipassana cycle** described in meditation literature:

```
1. Breath Control → 2. Mind Wandering → 3. Meta-Awareness → 4. Redirect Breath → [cycle repeats]
```

**Theoretical Foundation**: This cycle reflects the natural oscillation between focused attention, distraction detection, and attentional restoration that characterizes mindfulness meditation practice.

---

## Transition Decision Architecture

### Location in Code
- **Primary Implementation**: `act_inf_learner.py`, lines 756-886 (within `train()` method)
- **Theoretical Basis**: Implements concepts from Equation 4 (expected free energy for state transitions)

### Transition Decision Process

```python
# Handle state transitions
if current_dwell >= dwell_limit:
    # 1. Calculate transition probability based on free energy
    natural_prob = 0.8 + min(0.15, t / self.timesteps * 0.2)
    precision_factor = 1.5 if self.experience_level == 'expert' else 0.8
    fe_factor = min(0.3, free_energy * 0.3 * precision_factor)
    natural_prob = min(0.95, natural_prob + fe_factor)
    
    # 2. Determine if natural transition occurs
    if np.random.random() < natural_prob:
        # Apply state-specific transition logic
    else:
        # Fall back to sequential transition
```

### Free Energy Influence on Transitions

**Equation 4 Implementation**: The system implements concepts from:
```
P(s_{t+1}=s') ∝ exp(-β F_t(s')) × Θ(s_t → s')
```

Where:
- `P(s_{t+1}=s')`: Probability of transitioning to state s'
- `F_t(s')`: Free energy of potential target state
- `Θ(s_t → s')`: Threshold conditions for valid transitions
- `β`: Inverse temperature (precision parameter)

**Implementation Details**:
```python
# Higher free energy increases transition probability
precision_factor = 1.5 if self.experience_level == 'expert' else 0.8
fe_factor = min(0.3, free_energy * 0.3 * precision_factor)
natural_prob = min(0.95, natural_prob + fe_factor)
```

**Key Principles**:
1. **High Free Energy → Higher Transition Probability**: Systems with prediction errors are more likely to transition
2. **Expert Precision**: Experts have higher precision (β) in transition decisions
3. **Progressive Training**: Transition probability increases over simulation time
4. **Bounded Probability**: Natural transition probability capped at 95%

---

## State-Specific Transition Logic

### 1. Focused States → Mind Wandering

#### Transition Conditions
```python
# FOCUSED STATES TO MIND WANDERING
if current_state in ["breath_control", "redirect_breath"]:
    # Calculate combined distraction level
    distraction_level = (activations[self.thoughtseeds.index("pain_discomfort")] + 
                        activations[self.thoughtseeds.index("pending_tasks")])
    
    # Consider DMN/DAN ratio
    dmn_dan_ratio = network_acts['DMN'] / (network_acts['DAN'] + 0.1)
    
    # Either high distraction OR high DMN/DAN ratio can trigger transition
    if (distraction_level > self.transition_thresholds['mind_wandering'] or 
        dmn_dan_ratio > self.transition_thresholds['dmn_dan_ratio']):
        next_state = "mind_wandering"
        natural_transition = True
```

#### Threshold Parameters

| Experience Level | Distraction Threshold | DMN/DAN Ratio Threshold |
|------------------|----------------------|-------------------------|
| **Novice** | 0.6 | 0.5 |
| **Expert** | 0.7 | 0.6 |

**Interpretation**:
- **Distraction Threshold**: Combined activation of pain_discomfort + pending_tasks
- **DMN/DAN Ratio**: Default Mode vs Dorsal Attention Network balance
- **Expert Resilience**: Higher thresholds reflect greater resistance to distraction

#### Biological Mechanisms Modeled

1. **Distraction Accumulation**: Progressive buildup of competing thoughtseeds
2. **Network Competition**: DMN activation competing with DAN maintenance
3. **Threshold Crossing**: Discrete transition when capacity is exceeded
4. **Experience Modulation**: Expert practitioners require higher distraction levels

### 2. Mind Wandering → Meta-Awareness

#### Transition Conditions
```python
# MIND WANDERING TO META-AWARENESS
elif current_state == "mind_wandering":
    # Self-reflection is the key factor
    self_reflection = activations[self.thoughtseeds.index("self_reflection")]
    
    # Consider VAN activation as secondary factor
    van_activation = network_acts['VAN']
    
    # Simplified check with more accessible thresholds
    awareness_threshold = 0.35 if self.experience_level == 'expert' else 0.45
    
    if (self_reflection > awareness_threshold or 
        (van_activation > 0.4 and self_reflection > 0.3)):
        next_state = "meta_awareness"
        natural_transition = True
```

#### Detection Mechanisms

1. **Primary Pathway**: Self-reflection thoughtseed activation
   - **Expert Threshold**: 0.35 (more sensitive)
   - **Novice Threshold**: 0.45 (less sensitive)

2. **Secondary Pathway**: VAN (Ventral Attention Network) salience detection
   - **VAN Threshold**: 0.4
   - **Reduced Self-Reflection Requirement**: 0.3 when VAN is active

**Neuroscientific Basis**:
- **Self-Reflection**: Internal monitoring of mental states
- **VAN Activation**: Bottom-up salience detection of attentional lapses
- **Expert Sensitivity**: Lower thresholds reflect enhanced metacognitive awareness

### 3. Meta-Awareness → Focused States

#### Transition Conditions
```python
# META-AWARENESS TO FOCUSED STATES
elif current_state == "meta_awareness":
    # Base transition values (from activations)
    bf_value = activations[self.thoughtseeds.index("breath_focus")]
    eq_value = activations[self.thoughtseeds.index("equanimity")]
    
    # Network influences on transitions
    bf_value += network_acts['DAN'] * 0.2  # DAN enhances breath focus intention
    eq_value += network_acts['FPN'] * 0.2  # FPN enhances equanimity intention
    
    # Lower threshold for more reliable transitions
    threshold = self.transition_thresholds['return_focus']
    
    if bf_value > threshold and eq_value > threshold:
        # If both high, experts favor equanimity/redirect_breath
        if self.experience_level == 'expert' and eq_value > bf_value:
            next_state = "redirect_breath"
        else:
            next_state = "breath_control"
        natural_transition = True
    elif bf_value > threshold + 0.1:  # Higher certainty for single condition
        next_state = "breath_control"
        natural_transition = True
    elif eq_value > threshold + 0.1:  # Higher certainty for single condition
        next_state = "redirect_breath"
        natural_transition = True
```

#### Decision Logic

1. **Network Enhancement**: 
   - **DAN → Breath Focus**: Dorsal Attention Network enhances breath focus intention
   - **FPN → Equanimity**: Frontoparietal Network enhances equanimity intention

2. **Dual Condition Success**: Both breath_focus and equanimity above threshold
   - **Expert Preference**: Favor redirect_breath when equanimity > breath_focus
   - **Novice Default**: Generally return to breath_control

3. **Single Condition Success**: Higher certainty threshold (threshold + 0.1) for single pathway

4. **Return Focus Thresholds**:
   - **Expert**: 0.25 (easier return to focus)
   - **Novice**: 0.3 (more difficult return to focus)

---

## Transition Threshold System

### Threshold Configuration Classes

```python
@dataclass
class TransitionThresholds:
    """Thresholds for state transitions"""
    mind_wandering: float      # Distraction level threshold
    dmn_dan_ratio: float      # DMN/DAN ratio threshold
    meta_awareness: float     # Self-reflection threshold for meta-awareness
    return_focus: float       # Threshold to return to focused states
```

### Experience-Level Thresholds

```python
# Novice thresholds
TransitionThresholds(
    mind_wandering=0.6,    # Lower resistance to distraction
    dmn_dan_ratio=0.5,     # Lower DMN/DAN threshold
    meta_awareness=0.4,    # Higher threshold for awareness
    return_focus=0.3       # Higher threshold to return to focus
)

# Expert thresholds
TransitionThresholds(
    mind_wandering=0.7,    # Higher resistance to distraction
    dmn_dan_ratio=0.6,     # Higher DMN/DAN threshold
    meta_awareness=0.3,    # Lower threshold for awareness (more sensitive)
    return_focus=0.25      # Lower threshold to return to focus (easier)
)
```

### Threshold Interpretation

| Threshold Type | Novice | Expert | Interpretation |
|----------------|--------|--------|----------------|
| **mind_wandering** | 0.6 | 0.7 | Distraction resistance |
| **dmn_dan_ratio** | 0.5 | 0.6 | Network balance tolerance |
| **meta_awareness** | 0.4 | 0.3 | Metacognitive sensitivity |
| **return_focus** | 0.3 | 0.25 | Focus restoration ease |

**Key Patterns**:
1. **Expert Advantages**: Higher distraction resistance, easier focus restoration
2. **Expert Sensitivity**: Lower meta-awareness threshold (better detection)
3. **Novice Challenges**: More vulnerable to distraction, harder to return to focus

---

## Temporal Dynamics and Dwell Times

### Dwell Time Management

```python
def get_dwell_time(self, state):
    """Get state-specific dwell time based on experience level."""
    config_min, config_max = STATE_DWELL_TIMES[self.experience_level][state]
    return max(min_biological, min(max_biological, 
                                   np.random.randint(config_min, config_max + 1)))
```

### Experience-Level Dwell Patterns

| State | Novice Range (timesteps) | Expert Range (timesteps) | Pattern |
|-------|-------------------------|--------------------------|---------|
| **Breath Control** | 5-15 | 15-25 | Experts sustain longer |
| **Mind Wandering** | 15-30 | 8-12 | Experts recover faster |
| **Meta-Awareness** | 2-5 | 1-3 | Experts more efficient |
| **Redirect Breath** | 2-5 | 1-3 | Experts more skillful |

**Biological Constraints**:
- **Minimum Brief States**: 1 timestep for meta-awareness, redirect_breath
- **Minimum Sustained States**: 3 timesteps for breath_control, mind_wandering
- **Maximum Ranges**: Prevent unrealistic dwell times

### Dwell Time vs. Transition Probability

```python
if current_dwell >= dwell_limit:
    # Only check transitions when dwell time is reached
    # This prevents premature state switching
```

**Key Principles**:
1. **Minimum Dwell Enforcement**: States must persist for minimum biologically plausible duration
2. **Variable Duration**: Random sampling within configured ranges
3. **Experience Differentiation**: Experts show longer focused states, shorter distracted states

---

## Transition Smoothing and Biological Plausibility

### Gradual Activation Transitions

```python
# More gradual transition with biological variability
new_target = self.get_target_activations(current_state, meta_awareness)

# Add variability to target activations
for i in range(len(new_target)):
    variation = 1.0 + np.random.uniform(-0.05, 0.1)  # 5-10% variation
    new_target[i] *= variation
    new_target[i] = max(0.06, new_target[i])

# Conservative blending with variability
blend_factor = 0.4 * (1.0 + np.random.uniform(-0.1, 0.1))  # 36-44% blend
activations = (1 - blend_factor) * activations + blend_factor * new_target
```

### Multi-Timestep Transition Process

```python
# Add transition markers for continued smoothing
self.in_transition = True
self.transition_counter = 3 + np.random.randint(0, 2)  # Variable transition time
self.transition_target = new_target.copy()
```

**Continued Smoothing During Transition**:
```python
if hasattr(self, 'in_transition') and self.in_transition:
    blend_factor = 0.3 * (1.0 + np.random.uniform(-0.1, 0.1))  # 27-33% blend
    
    # Add small random perturbations to transition target
    perturbed_target = self.transition_target.copy()
    perturbed_target += np.random.normal(0, 0.02, size=len(perturbed_target))
    perturbed_target = np.clip(perturbed_target, 0.05, 1.0)
    
    # Apply blending with perturbed target
    activations = (1 - blend_factor) * activations + blend_factor * perturbed_target
    
    # Decrement counter and check if transition is complete
    self.transition_counter -= 1
    if self.transition_counter <= 0:
        self.in_transition = False
```

**Biological Plausibility Features**:
1. **Multi-Timestep Transitions**: Transitions occur over 3-5 timesteps, not instantaneously
2. **Variable Blending**: Random variation in transition speed
3. **Target Perturbation**: Small noise added to prevent rigid transitions
4. **Activation Bounds**: All values constrained to biologically plausible ranges

---

## Transition Statistics and Analysis

### Transition Counting System

```python
# Record the transition
self.transition_counts[current_state][next_state] += 1

# Track transition type
if natural_transition:
    self.natural_transition_count += 1
else:
    self.forced_transition_count += 1
```

### Transition Matrix Generation

The system generates comprehensive transition statistics:

```python
transition_stats = {
    'transition_counts': self.transition_counts,
    'transition_thresholds': self.transition_thresholds,
    'natural_transitions': self.natural_transition_count,
    'forced_transitions': self.forced_transition_count,
    'transition_timestamps': transition_timestamps,
    'state_transition_patterns': state_transition_patterns
}
```

### Natural vs. Forced Transitions

**Natural Transitions**: Driven by model dynamics (free energy, threshold crossing)
**Forced Transitions**: Sequential fallback when natural transitions don't occur

**Quality Assurance**:
```python
if self.natural_transition_count < 4:  # Require at least 4 natural transitions
    print(f"WARNING: Only {self.natural_transition_count} natural transitions occurred.")
    # Add additional natural transitions if needed
```

### Transition Pattern Storage

```python
state_transition_patterns.append((
    current_state,           # Source state
    next_state,             # Target state
    {ts: activations[i] for i, ts in enumerate(self.thoughtseeds)},  # Thoughtseed pattern
    {net: val for net, val in network_acts.items()},                # Network pattern
    free_energy             # Free energy at transition
))
```

This comprehensive data enables analysis of:
- **Transition Triggers**: What conditions lead to specific transitions
- **Network States**: Network activation patterns during transitions
- **Free Energy Dynamics**: Energy landscape during state changes

---

## Experience-Level Transition Differences

### Expert Transition Characteristics

1. **Enhanced Stability**: 
   - Higher thresholds for mind_wandering (0.7 vs 0.6)
   - Longer dwell times in focused states (15-25 vs 5-15)

2. **Improved Detection**:
   - Lower meta_awareness threshold (0.3 vs 0.4)
   - More sensitive to attentional lapses

3. **Efficient Recovery**:
   - Lower return_focus threshold (0.25 vs 0.3)
   - Shorter transition states (1-3 vs 2-5)

4. **Strategic Flexibility**:
   - Preference for redirect_breath when equanimity is high
   - Network-enhanced transition decisions

### Novice Transition Characteristics

1. **Reduced Stability**:
   - Lower distraction resistance
   - Shorter focused state durations
   - Longer mind-wandering episodes

2. **Detection Challenges**:
   - Higher meta_awareness threshold
   - Less sensitive to subtle attentional shifts

3. **Recovery Difficulties**:
   - Higher barriers to return to focus
   - Longer transition durations
   - More forced transitions

---

## Validation and Quality Assurance

### Transition Validation Checks

```python
# Ensure reasonable transition patterns
assert self.natural_transition_count >= 4, "Insufficient natural transitions"

# Validate transition counts
total_transitions = sum(sum(counts.values()) for counts in self.transition_counts.values())
assert total_transitions > 0, "No transitions recorded"

# Check for unrealistic patterns
for state in self.states:
    self_transitions = self.transition_counts[state][state]
    assert self_transitions == 0, f"Self-transitions detected in {state}"
```

### Biological Plausibility Checks

```python
# Ensure minimum dwell times
assert all(dwell >= 1 for dwell in dwell_times), "Dwell times below biological minimum"

# Validate threshold ranges
for threshold in self.transition_thresholds.__dict__.values():
    assert 0.0 <= threshold <= 1.0, "Threshold outside valid range"
```

### Experience-Level Validation

```python
def validate_expert_novice_differences():
    """Ensure expert advantages are maintained"""
    expert_thresholds = TransitionThresholds.expert()
    novice_thresholds = TransitionThresholds.novice()
    
    # Experts should have higher distraction resistance
    assert expert_thresholds.mind_wandering > novice_thresholds.mind_wandering
    
    # Experts should have lower meta-awareness thresholds (more sensitive)
    assert expert_thresholds.meta_awareness < novice_thresholds.meta_awareness
    
    # Experts should have easier return to focus
    assert expert_thresholds.return_focus < novice_thresholds.return_focus
```

---

## Future Enhancements

### Advanced Transition Models

1. **Probabilistic State Models**: Full Markov chain implementation with learned transition probabilities
2. **Hierarchical Transitions**: Multi-level state transitions (e.g., sub-states within mind_wandering)
3. **Context-Dependent Thresholds**: Dynamic thresholds based on meditation session progress
4. **Individual Differences**: Personalized threshold learning based on practitioner characteristics

### Enhanced Free Energy Integration

1. **Prospective Free Energy**: Compute expected free energy for all possible next states
2. **Policy Selection**: Choose transitions that minimize expected future free energy
3. **Uncertainty Estimation**: Model uncertainty in state predictions
4. **Adaptive Precision**: Learn precision parameters from experience

### Empirical Validation

1. **EEG Validation**: Compare transition patterns with empirical EEG data
2. **Behavioral Validation**: Validate against self-reported meditation experiences
3. **Parameter Fitting**: Optimize thresholds based on empirical data
4. **Cross-Validation**: Test generalization across different meditation styles

---

*This documentation covers the comprehensive state transition system that governs the temporal dynamics of meditation states in the Active Inference framework. The system implements biologically plausible transition mechanisms while maintaining theoretical consistency with free energy minimization principles.*
