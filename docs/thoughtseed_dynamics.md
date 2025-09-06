# Thoughtseed Dynamics Documentation

## Overview

This document provides comprehensive technical details on thoughtseed dynamics—the agent-like attentional entities that compete for mental workspace in the meditation simulation framework. Thoughtseeds represent transient, competing cognitive processes that form Markov blankets and engage in predictive processing to minimize variational free energy through bidirectional coupling with attentional networks.

---

## Theoretical Foundation

### Conceptual Framework

**Thoughtseeds** are conceptualized as:
- **Agent-like Entities**: Autonomous cognitive processes with their own dynamics
- **Markov Blankets**: Self-organizing boundaries that separate internal states from external environment
- **Competitive Processes**: Multiple thoughtseeds compete for limited attentional resources
- **Predictive Processors**: Each thoughtseed attempts to minimize its own prediction errors

### Active Inference Integration

Thoughtseeds implement Active Inference through:
1. **Generative Models**: Each thoughtseed maintains predictions about sensory input
2. **Variational Free Energy**: Thoughtseeds minimize prediction errors and complexity
3. **Action Selection**: Thoughtseeds influence attention allocation to reduce uncertainty
4. **Learning**: Thoughtseed-network mappings adapt based on prediction errors

---

## Five-Thoughtseed Architecture

### Core Thoughtseed Definitions

| Thoughtseed | Category | Function | Experience Difference |
|------------|----------|----------|---------------------|
| **breath_focus** | Focus | Sustained attention on breathing | Experts: stronger, more stable |
| **pain_discomfort** | Distraction | Bodily sensations, discomfort | Experts: less disruptive |
| **pending_tasks** | Distraction | Mental to-do items, planning | Experts: greatly reduced |
| **self_reflection** | Metacognition | Introspective awareness | Experts: more skillful |
| **equanimity** | Regulation | Emotional balance, acceptance | Experts: much stronger |

### Thoughtseed Agent Configuration

```python
THOUGHTSEED_AGENTS = {
    "breath_focus": ThoughtseedAgent(
        id=0,
        category="focus",
        intentional_weights={"novice": 0.8, "expert": 0.95},
        decay_rate=0.005,
        recovery_rate=0.06
    ),
    "pain_discomfort": ThoughtseedAgent(
        id=1,
        category="distraction",
        intentional_weights={"novice": 0.4, "expert": 0.6},
        decay_rate=0.003,
        recovery_rate=0.05
    ),
    "pending_tasks": ThoughtseedAgent(
        id=2,
        category="distraction",
        intentional_weights={"novice": 0.3, "expert": 0.5},
        decay_rate=0.002,
        recovery_rate=0.03
    ),
    "self_reflection": ThoughtseedAgent(
        id=3,
        category="metacognition",
        intentional_weights={"novice": 0.5, "expert": 0.8},
        decay_rate=0.004,
        recovery_rate=0.04
    ),
    "equanimity": ThoughtseedAgent(
        id=4,
        category="regulation",
        intentional_weights={"novice": 0.5, "expert": 0.9},
        decay_rate=0.001,
        recovery_rate=0.02
    )
}
```

### Key Parameters

**Intentional Weights**: Represent the maximum sustainable activation level for each thoughtseed
- **Expert Advantages**: Higher breath_focus (0.95 vs 0.8), equanimity (0.9 vs 0.5)
- **Expert Regulation**: Higher self_reflection (0.8 vs 0.5), better pain tolerance (0.6 vs 0.4)
- **Novice Challenges**: Lower focus stability, greater distraction susceptibility

**Decay Rates**: Natural attenuation when not actively maintained
- **Fastest Decay**: breath_focus (0.005) - requires active maintenance
- **Moderate Decay**: self_reflection (0.004), pain_discomfort (0.003)  
- **Slow Decay**: pending_tasks (0.002), equanimity (0.001) - more persistent

**Recovery Rates**: Speed of activation increase when stimulated
- **Fastest Recovery**: breath_focus (0.06) - can be rapidly engaged
- **Moderate Recovery**: pain_discomfort (0.05), self_reflection (0.04)
- **Slow Recovery**: pending_tasks (0.03), equanimity (0.02) - develop gradually

---

## Target Activation System

### State-Dependent Activation Patterns

#### Base Activation Profiles

```python
BASE_ACTIVATIONS = {
    "breath_control": StateTargetActivations(
        breath_focus=0.7,      # High - primary focus object
        equanimity=0.3,        # Moderate - supportive regulation
        pain_discomfort=0.15,  # Low - peripheral distraction
        pending_tasks=0.1,     # Low - suppressed planning
        self_reflection=0.2    # Low-moderate - background awareness
    ),
    "mind_wandering": StateTargetActivations(
        breath_focus=0.1,      # Very low - attention drifted away
        equanimity=0.1,        # Low - reduced emotional regulation
        pain_discomfort=0.6,   # High - amplified bodily sensations
        pending_tasks=0.7,     # High - dominant mental activity
        self_reflection=0.1    # Low - reduced self-monitoring
    ),
    "meta_awareness": StateTargetActivations(
        breath_focus=0.2,      # Low - not yet refocused
        equanimity=0.3,        # Moderate - emotional balance
        pain_discomfort=0.15,  # Low - reduced salience
        pending_tasks=0.15,    # Low - reduced task focus
        self_reflection=0.8    # Very high - dominant introspection
    ),
    "redirect_breath": StateTargetActivations(
        breath_focus=0.6,      # High - returning to focus
        equanimity=0.7,        # Very high - skillful regulation
        pain_discomfort=0.2,   # Low - managed distraction
        pending_tasks=0.1,     # Low - suppressed planning
        self_reflection=0.4    # Moderate - ongoing monitoring
    )
}
```

### Meta-Awareness Modulation

Thoughtseed activations are dynamically modulated by current meta-awareness levels:

```python
META_AWARENESS_MODULATORS = {
    "breath_control": {
        "breath_focus": 0.1,       # Meta-awareness enhances focus
        "equanimity": 0.2,         # Meta-awareness enhances regulation
        "pain_discomfort": 0.0,    # No direct effect
        "pending_tasks": 0.0,      # No direct effect
        "self_reflection": 0.1     # Slight enhancement
    },
    "mind_wandering": {
        "breath_focus": 0.0,       # No enhancement during wandering
        "equanimity": 0.0,         # No enhancement during wandering
        "pain_discomfort": -0.1,   # Meta-awareness reduces pain focus
        "pending_tasks": -0.1,     # Meta-awareness reduces task focus
        "self_reflection": 0.3     # Strong enhancement for detection
    }
}
```

**Key Patterns**:
- **Focus Enhancement**: Meta-awareness strengthens breath_focus and equanimity during focused states
- **Distraction Reduction**: Meta-awareness reduces pain and task focus during mind-wandering
- **Detection Amplification**: Meta-awareness dramatically increases self_reflection during wandering

### Expert Adjustments

```python
EXPERT_ADJUSTMENTS = {
    "breath_control": {
        "breath_focus": 0.1,       # 10% stronger breath focus
        "equanimity": 0.15,        # 15% stronger equanimity
        "pain_discomfort": 0.0,    # No change
        "pending_tasks": 0.0,      # No change
        "self_reflection": 0.0     # No change
    },
    "mind_wandering": {
        "breath_focus": 0.0,       # No change
        "equanimity": 0.0,         # No change  
        "pain_discomfort": -0.1,   # 10% less pain distraction
        "pending_tasks": -0.25,    # 25% less task distraction
        "self_reflection": 0.0     # No change
    }
}
```

**Expert Advantages**:
- **Enhanced Focus**: Stronger breath focus and equanimity during practice
- **Reduced Distraction**: Significantly less mental task interference (25% reduction)
- **Pain Management**: Better tolerance for physical discomfort (10% reduction)

---

## Thoughtseed Activation Computation

### Target Activation Generation

**Location**: `rules_based_learner.py`, `get_target_activations()` method

```python
def get_target_activations(self, state, meta_awareness):
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

### Parameter Calculation Process

```python
@staticmethod
def get_target_activations(state, meta_awareness, experience_level='novice'):
    # 1. Start with base activations for this state
    activations = ThoughtseedParams.BASE_ACTIVATIONS[state].copy()
    
    # 2. Apply meta-awareness modulation
    for ts in activations:
        modulator = ThoughtseedParams.META_AWARENESS_MODULATORS[state][ts]
        activations[ts] += modulator * meta_awareness
    
    # 3. Apply expert adjustments if applicable
    if experience_level == 'expert':
        for ts in activations:
            activations[ts] += ThoughtseedParams.EXPERT_ADJUSTMENTS[state].get(ts, 0)
    
    return activations
```

**Computational Steps**:
1. **Base Pattern**: Load state-specific activation template
2. **Meta-Awareness Scaling**: Apply awareness-dependent modulation
3. **Experience Adjustment**: Add expert/novice differential effects
4. **Noise Addition**: Include biological variability
5. **Bound Enforcement**: Constrain to biologically plausible range [0.05, 1.0]

---

## Network-Thoughtseed Bidirectional Coupling

### Bottom-Up Influence: Thoughtseeds → Networks

**Location**: `act_inf_learner.py`, `compute_network_activations()` method

```python
# Calculate bottom-up influence: thoughtseeds -> networks
for i, ts in enumerate(self.thoughtseeds):
    ts_act = thoughtseed_activations[i]
    for net in self.networks:
        # Weighted contribution based on thoughtseed activation
        network_acts[net] += ts_act * ts_to_network[ts][net] * bottom_up_weight
```

#### Thoughtseed-to-Network Mapping

```python
"thoughtseed_contributions": {
    "breath_focus": NetworkProfile(DMN=0.2, VAN=0.3, DAN=0.8, FPN=0.6),
    "pain_discomfort": NetworkProfile(DMN=0.5, VAN=0.7, DAN=0.3, FPN=0.4),
    "pending_tasks": NetworkProfile(DMN=0.8, VAN=0.5, DAN=0.2, FPN=0.4),
    "self_reflection": NetworkProfile(DMN=0.6, VAN=0.4, DAN=0.3, FPN=0.8),
    "equanimity": NetworkProfile(DMN=0.3, VAN=0.3, DAN=0.5, FPN=0.9)
}
```

**Mapping Patterns**:
- **breath_focus**: Strongly activates DAN (0.8), moderately activates FPN (0.6)
- **pain_discomfort**: Strongly activates VAN (0.7), moderately activates DMN (0.5)
- **pending_tasks**: Dominates DMN (0.8), suppresses DAN (0.2)
- **self_reflection**: Strongly activates FPN (0.8), moderately activates DMN (0.6)
- **equanimity**: Dominates FPN (0.9), balanced across other networks

### Top-Down Influence: Networks → Thoughtseeds

**Location**: `act_inf_learner.py`, `network_modulated_activations()` method

```python
def network_modulated_activations(self, activations, network_acts, current_state):
    modulated_acts = activations.copy()
    
    # DMN enhances pending_tasks and self_reflection, suppresses breath_focus
    dmn_strength = network_acts['DMN']
    modulated_acts[self.thoughtseeds.index('pending_tasks')] += dmn_pending_value * dmn_strength
    modulated_acts[self.thoughtseeds.index('self_reflection')] += dmn_reflection_value * dmn_strength
    modulated_acts[self.thoughtseeds.index('breath_focus')] -= dmn_breath_value * dmn_strength

    # VAN enhances pain_discomfort (salience) and self_reflection during meta_awareness
    van_strength = network_acts['VAN']
    modulated_acts[self.thoughtseeds.index('pain_discomfort')] += van_pain_value * van_strength
    if current_state == "meta_awareness":
        modulated_acts[self.thoughtseeds.index('self_reflection')] += van_reflection_value * van_strength

    # DAN enhances breath_focus, suppresses distractions
    dan_strength = network_acts['DAN']
    modulated_acts[self.thoughtseeds.index('breath_focus')] += dan_breath_value * dan_strength
    modulated_acts[self.thoughtseeds.index('pending_tasks')] -= dan_pending_value * dan_strength
    modulated_acts[self.thoughtseeds.index('pain_discomfort')] -= dan_pain_value * dan_strength
    
    # FPN enhances self_reflection and equanimity (metacognition and regulation)
    fpn_strength = network_acts['FPN']
    modulated_acts[self.thoughtseeds.index('self_reflection')] += fpn_reflection_value * fpn_strength
    modulated_acts[self.thoughtseeds.index('equanimity')] += fpn_equanimity_value * fpn_strength
```

#### Network Modulation Effects

| Network | Enhances | Suppresses | Function |
|---------|----------|------------|----------|
| **DMN** | pending_tasks, self_reflection | breath_focus | Default mode processing |
| **VAN** | pain_discomfort, (self_reflection in meta-awareness) | - | Salience detection |
| **DAN** | breath_focus | pending_tasks, pain_discomfort | Focused attention |
| **FPN** | self_reflection, equanimity | - | Executive control |

**Experience-Level Modulation**:
- **Expert FPN Enhancement**: 1.2x multiplier for experts (vs 1.0x for novices)
- **Stronger Regulation**: Experts get enhanced equanimity and self_reflection from FPN
- **Better Focus**: Experts get stronger breath_focus enhancement from DAN

---

## Competitive Dynamics and Resource Allocation

### Attention Competition Model

Thoughtseeds compete for limited attentional resources through:

1. **Activation Normalization**: 
   ```python
   modulated_acts = np.clip(modulated_acts, 0.05, 1.0)
   ```

2. **Winner-Take-More Dynamics**: Dominant thoughtseeds suppress competitors indirectly through network influences

3. **Resource Constraints**: High activation in one thoughtseed can drain resources from others

### Distraction Buildup Mechanism

**Location**: `act_inf_learner.py`, training loop distraction handling

```python
# Handle distraction growth in focused states
if current_state in ["breath_control", "redirect_breath"]:
    time_in_focused_state += 1
    dwell_factor = min(1.0, current_dwell / max(10, dwell_limit))
    
    distraction_scale = 2.5 if self.experience_level == 'novice' else 1.2
    distraction_growth = 0.035 * dwell_factor * distraction_scale
    
    # Random boost events
    boost_chance = 0.1
    boost_factor = 1.0
    if np.random.random() < boost_chance:
        boost_factor = 3.0
    
    for i, ts in enumerate(self.thoughtseeds):
        if ts in ["pain_discomfort", "pending_tasks"]:
            activations[i] += distraction_growth * boost_factor
```

**Distraction Dynamics**:
- **Progressive Buildup**: Distraction increases with time spent in focused states
- **Experience Scaling**: Novices experience 2.1x stronger distraction buildup
- **Random Boosting**: 10% chance of 3x distraction spike (representing external triggers)
- **Selective Targeting**: Only affects distraction thoughtseeds (pain, tasks)

### Focus Fatigue Mechanism

```python
# Breath focus fatigue
for i, ts in enumerate(self.thoughtseeds):
    if ts == "breath_focus":
        fatigue_rate = 0.005 if self.experience_level == 'expert' else 0.01
        fatigue = fatigue_rate * dwell_factor * time_in_focused_state/10
        activations[i] = max(0.2, activations[i] - fatigue)
```

**Fatigue Features**:
- **Gradual Decline**: Breath focus slowly decreases with sustained attention
- **Expert Resilience**: Experts experience 50% less fatigue (0.005 vs 0.01)
- **Minimum Baseline**: Focus cannot drop below 0.2 (maintaining some attention)
- **Time Dependency**: Fatigue increases with duration in focused state

---

## Experience-Level Thoughtseed Differences

### Novice Characteristics

#### Enhanced Mind-Wandering

```python
if current_state == "mind_wandering" and self.experience_level == 'novice':
    # Make pending_tasks more dominant and persistent
    pt_idx = self.thoughtseeds.index("pending_tasks")
    activations[pt_idx] *= 1.15  # 15% boost
    
    # Create interference with self-reflection
    sr_idx = self.thoughtseeds.index("self_reflection")
    interference = min(0.3, activations[pt_idx] * 0.4)
    activations[sr_idx] = max(0.05, activations[sr_idx] - interference)
```

**Novice Mind-Wandering Patterns**:
- **Dominant Task Focus**: pending_tasks gets 15% boost during mind-wandering
- **Metacognitive Interference**: High task focus reduces self_reflection ability
- **Reduced Detection**: Lower meta_awareness makes noticing wandering harder

#### Weaker Regulatory Capacity

```python
# Novice intentional weights
"equanimity": {"novice": 0.5, "expert": 0.9}  # 44% lower max capacity
"self_reflection": {"novice": 0.5, "expert": 0.8}  # 38% lower capacity
```

### Expert Characteristics

#### Enhanced Focus Synergies

```python
if self.experience_level == 'expert' and current_state in ["redirect_breath", "meta_awareness"]:
    bf_idx = self.thoughtseeds.index("breath_focus")
    eq_idx = self.thoughtseeds.index("equanimity")
    
    if activations[bf_idx] > 0.3 and activations[eq_idx] > 0.3:
        # Synergistic boost when both are moderately active
        boost = 0.03 * min(activations[bf_idx], activations[eq_idx])
        activations[bf_idx] += boost
        activations[eq_idx] += boost
```

**Expert Synergies**:
- **Breath-Equanimity Coupling**: Mutual reinforcement when both are active
- **Pain Regulation**: Higher equanimity suppresses pain_discomfort
- **Strategic Transitions**: Preferential use of redirect_breath over breath_control

#### Reduced Distraction Susceptibility

```python
# Expert adjustments during mind-wandering
"pending_tasks": -0.25,    # 25% reduction in task distraction
"pain_discomfort": -0.1,   # 10% reduction in pain distraction
```

**Expert Advantages**:
- **Task Management**: 25% less mental task interference
- **Pain Tolerance**: 10% better discomfort management
- **Faster Recovery**: Shorter mind-wandering episodes

---

## Thoughtseed Learning and Adaptation

### Network Profile Learning

**Location**: `act_inf_learner.py`, `update_network_profiles()` method

```python
def update_network_profiles(self, thoughtseed_activations, network_activations, 
                           current_state, prediction_errors):
    # For each thoughtseed contribution to networks
    for i, ts in enumerate(self.thoughtseeds):
        ts_act = thoughtseed_activations[i]  # z_i(t) in Equation 3
        
        # Only update when thoughtseed is significantly active
        if ts_act > 0.2:
            for net in self.networks:
                current_error = prediction_errors[net]  # δ_k(t) in Equation 3
                
                # Calculate precision (confidence)
                precision = 1.0 + (5.0 if self.experience_level == 'expert' else 2.0) * len(self.network_activations_history)/self.timesteps
                
                # Bayesian-inspired update (approximating Equation 3)
                error_sign = 1 if network_activations[net] < expected[net] else -1
                update = self.learning_rate * (error_sign * current_error) * ts_act / precision
                
                # Update contribution - implements W_ik ← W_ik + update
                self.learned_network_profiles["thoughtseed_contributions"][ts][net] += update
                
                # Ensure biological plausibility by bounding weights [0.1, 0.9]
                self.learned_network_profiles["thoughtseed_contributions"][ts][net] = np.clip(
                    self.learned_network_profiles["thoughtseed_contributions"][ts][net], 0.1, 0.9)
```

**Learning Mechanisms**:
1. **Activity-Dependent Updates**: Only active thoughtseeds (>0.2) update their profiles
2. **Error-Driven Learning**: Updates proportional to network prediction errors
3. **Experience-Dependent Precision**: Experts have higher confidence in updates
4. **Bounded Weights**: Maintains biological plausibility [0.1, 0.9]

### Thoughtseed-Network Mapping Evolution

Initial thoughtseed-network mappings evolve through training:

**Example Evolution** (breath_focus → DAN):
```
Initial: 0.8 (configuration-based)
After Training (Novice): 0.75 ± 0.05 (some degradation due to inconsistency)
After Training (Expert): 0.85 ± 0.02 (strengthened through consistent practice)
```

**Learning Patterns**:
- **Expert Strengthening**: Consistent practice strengthens functional mappings
- **Novice Variability**: Inconsistent patterns lead to more variable mappings
- **Specialization**: Frequently co-active thoughtseed-network pairs strengthen

---

## Biological Plausibility and Validation

### Activation Range Constraints

```python
# Ensure valid range after all modulations
activations = np.clip(activations, 0.05, 1.0)
```

**Range Interpretation**:
- **0.05**: Minimal background activation (not completely absent)
- **1.0**: Maximum sustainable activation (resource-limited)
- **Typical Range**: Most activations fall between 0.1-0.8 during normal operation

### Physiological Noise Addition

```python
# Add physiological noise to all activations
for i, ts in enumerate(self.thoughtseeds):
    noise_level = 0.005  # Base noise
    
    # More noise during mind_wandering
    if current_state == "mind_wandering":
        noise_level = 0.015
    
    # Different thoughtseeds have different noise characteristics
    if ts in ["breath_focus", "equanimity"]:
        noise_level *= 0.8  # More stable
    elif ts in ["pain_discomfort"]:
        noise_level *= 1.5  # More variable
    
    # Apply the noise
    activations[i] += np.random.normal(0, noise_level)
```

**Noise Characteristics**:
- **State-Dependent**: Higher noise during mind-wandering (reflecting instability)
- **Thoughtseed-Specific**: Focus and equanimity more stable, pain more variable
- **Gaussian Distribution**: Biologically plausible noise model

### Extreme Value Prevention

```python
# Cap extreme values for specific thoughtseeds
for i, ts in enumerate(self.thoughtseeds):
    if ts == "pending_tasks" and activations[i] > 0.8:
        activations[i] = 0.8  # Prevent unrealistic task obsession
```

---

## Dominant Thoughtseed Analysis

### Dominant Thoughtseed Identification

```python
# Identify dominant thoughtseed
dominant_ts = self.thoughtseeds[np.argmax(activations)]
self.dominant_ts_history.append(dominant_ts)
```

### Typical Dominance Patterns

| State | Novice Dominant | Expert Dominant | Pattern Difference |
|-------|----------------|-----------------|-------------------|
| **Breath Control** | breath_focus | breath_focus | Similar patterns |
| **Mind Wandering** | pending_tasks | pending_tasks | Expert: shorter duration |
| **Meta-Awareness** | self_reflection | self_reflection | Expert: more efficient |
| **Redirect Breath** | breath_focus | equanimity | Expert preference for regulation |

**Key Observations**:
- **Expert Strategic Preference**: Experts favor equanimity during redirect_breath
- **Novice Task Dominance**: Novices show stronger pending_tasks during wandering
- **Stability Differences**: Experts show more consistent patterns

### Transition Triggers

Dominant thoughtseed changes often trigger state transitions:
- **pending_tasks dominance** → mind_wandering state
- **self_reflection dominance** → meta_awareness state  
- **breath_focus/equanimity dominance** → focused states

---

## Future Enhancements

### Advanced Thoughtseed Models

1. **Hierarchical Thoughtseeds**: Sub-thoughtseeds within each main category
2. **Temporal Dependencies**: Thoughtseed activations influenced by recent history
3. **Individual Differences**: Personalized thoughtseed profiles
4. **Cultural Variations**: Different thoughtseed configurations for different meditation traditions

### Enhanced Competition Models

1. **Resource Allocation**: Explicit modeling of limited attentional resources
2. **Coalition Formation**: Thoughtseeds forming temporary alliances
3. **Inhibitory Networks**: Explicit inhibitory connections between competing thoughtseeds
4. **Energy Dynamics**: Thoughtseed activation costs and energy depletion

### Empirical Validation

1. **EEG Correlation**: Validate thoughtseed activations against neural markers
2. **Experience Sampling**: Compare with real-time meditation reports
3. **Physiological Measures**: Correlate with heart rate, skin conductance, etc.
4. **Longitudinal Studies**: Track thoughtseed pattern changes with meditation training

---

*This documentation covers the comprehensive thoughtseed dynamics system that models the competitive attentional processes underlying meditation practice. The agent-like thoughtseeds provide a biologically plausible foundation for understanding how different cognitive processes compete for and coordinate attentional resources in contemplative states.*
