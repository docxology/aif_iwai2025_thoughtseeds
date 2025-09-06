# Configuration System Documentation

## Overview

This document provides comprehensive technical details on the configuration system that manages all parameters, dataclasses, and experience-level specific settings in the meditation simulation framework. The configuration system in `meditation_config.py` provides type-safe, modular parameter management using Python dataclasses.

---

## Core Configuration Architecture

### Module Structure
- **File**: `meditation_config.py`
- **Primary Approach**: Dataclass-based configuration management
- **Key Features**: Type safety, experience-level specialization, literature-based defaults

### Base Constants

```python
# Core thoughtseed and state definitions
THOUGHTSEEDS = ['breath_focus', 'pain_discomfort', 'pending_tasks', 'self_reflection', 'equanimity']
STATES = ['breath_control', 'mind_wandering', 'meta_awareness', 'redirect_breath']
```

These core constants define the fundamental entities in the meditation simulation framework.

---

## Effect Magnitude System

### EffectMagnitude Dataclass

```python
@dataclass
class EffectMagnitude:
    """Standardized effect magnitudes for model adjustments"""
    NONE: float = 0.0
    WEAK: float = 0.1
    MODERATE: float = 0.2
    STRONG: float = 0.3
```

This standardized system ensures consistent effect sizes across all model parameters and prevents arbitrary parameter values.

### ContextualEffect System

```python
@dataclass
class ContextualEffect:
    """Effect that varies by meditation state and practitioner experience"""
    base_value: float
    state_modifiers: Dict[str, float] = field(default_factory=dict)
    experience_modifiers: Dict[str, float] = field(default_factory=dict)
    
    def get_value(self, state, experience_level):
        """Calculate contextual value based on state and experience"""
        value = self.base_value
        value += self.state_modifiers.get(state, 0.0)
        value += self.experience_modifiers.get(experience_level, 0.0)
        return value
```

**Key Features**:
- **Base Value**: Default effect magnitude
- **State Modifiers**: State-specific adjustments
- **Experience Modifiers**: Experience-level adjustments
- **Dynamic Calculation**: Runtime parameter computation

**Usage Example**:
```python
dmn_boost = ContextualEffect(
    base_value=EffectMagnitude.MODERATE,
    state_modifiers={"mind_wandering": EffectMagnitude.WEAK},
    experience_modifiers={"expert": -EffectMagnitude.WEAK}
)

# Runtime calculation
boost_value = dmn_boost.get_value("mind_wandering", "expert")
# Result: 0.2 + 0.1 + (-0.1) = 0.2
```

---

## State Dwell Time Configuration

### DwellTimeConfig Dataclass

```python
@dataclass
class DwellTimeConfig:
    """State dwell time configuration"""
    breath_control: Tuple[int, int]
    mind_wandering: Tuple[int, int]
    meta_awareness: Tuple[int, int]
    redirect_breath: Tuple[int, int]
```

### Experience-Level Defaults

```python
@classmethod
def novice(cls) -> 'DwellTimeConfig':
    return cls(
        breath_control=(5, 15),    # Shorter sustained attention
        mind_wandering=(15, 30),   # Longer mind-wandering episodes
        meta_awareness=(2, 5),     # Brief meta-awareness
        redirect_breath=(2, 5)     # Brief redirection
    )

@classmethod
def expert(cls) -> 'DwellTimeConfig':
    return cls(
        breath_control=(15, 25),   # Longer sustained attention
        mind_wandering=(8, 12),    # Shorter mind-wandering episodes
        meta_awareness=(1, 3),     # Very brief, efficient meta-awareness
        redirect_breath=(1, 3)     # Very brief, efficient redirection
    )
```

**Key Patterns**:
- **Experts**: Longer focused states, shorter distracted states
- **Novices**: Shorter focused states, longer distracted states
- **Biological Bounds**: All dwell times are biologically plausible ranges

---

## Thoughtseed Agent Configuration

### ThoughtseedAgent Dataclass

```python
@dataclass
class ThoughtseedAgent:
    """Configuration for a single thoughtseed agent"""
    id: int
    category: str
    intentional_weights: Dict[str, float]
    decay_rate: float
    recovery_rate: float
```

### Thoughtseed Agent Definitions

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

### Thoughtseed Categories

| Category | Thoughtseeds | Function | Expert vs Novice |
|----------|--------------|----------|------------------|
| **Focus** | breath_focus | Sustained attention | Experts: 95% vs Novices: 80% |
| **Distraction** | pain_discomfort, pending_tasks | Mind-wandering triggers | Experts more stable |
| **Metacognition** | self_reflection | Meta-awareness generation | Experts: 80% vs Novices: 50% |
| **Regulation** | equanimity | Emotional regulation | Experts: 90% vs Novices: 50% |

---

## Network Profile System

### NetworkProfile Dataclass

```python
@dataclass
class NetworkProfile:
    """Network profiles for thoughtseeds and states"""
    DMN: float
    VAN: float
    DAN: float
    FPN: float
```

### Literature-Based Network Profiles

#### Thoughtseed-to-Network Contributions

```python
"thoughtseed_contributions": {
    "breath_focus": NetworkProfile(DMN=0.2, VAN=0.3, DAN=0.8, FPN=0.6),
    "pain_discomfort": NetworkProfile(DMN=0.5, VAN=0.7, DAN=0.3, FPN=0.4),
    "pending_tasks": NetworkProfile(DMN=0.8, VAN=0.5, DAN=0.2, FPN=0.4),
    "self_reflection": NetworkProfile(DMN=0.6, VAN=0.4, DAN=0.3, FPN=0.8),
    "equanimity": NetworkProfile(DMN=0.3, VAN=0.3, DAN=0.5, FPN=0.9)
}
```

**Profile Patterns**:
- **breath_focus**: High DAN (attention), moderate FPN (control)
- **pain_discomfort**: High VAN (salience), moderate DMN (self-referential)
- **pending_tasks**: High DMN (default mode), low DAN (attention disruption)
- **self_reflection**: High FPN (executive), moderate DMN (introspection)
- **equanimity**: High FPN (regulation), balanced other networks

#### State Network Expectations

```python
"state_expected_profiles": {
    "breath_control": {
        "novice": NetworkProfile(DMN=0.35, VAN=0.4, DAN=0.7, FPN=0.5),
        "expert": NetworkProfile(DMN=0.2, VAN=0.4, DAN=0.85, FPN=0.7)
    },
    "mind_wandering": {
        "novice": NetworkProfile(DMN=0.85, VAN=0.45, DAN=0.2, FPN=0.35),
        "expert": NetworkProfile(DMN=0.65, VAN=0.5, DAN=0.35, FPN=0.55)
    },
    "meta_awareness": {
        "novice": NetworkProfile(DMN=0.35, VAN=0.7, DAN=0.5, FPN=0.45),
        "expert": NetworkProfile(DMN=0.3, VAN=0.8, DAN=0.6, FPN=0.6)
    },
    "redirect_breath": {
        "novice": NetworkProfile(DMN=0.3, VAN=0.45, DAN=0.65, FPN=0.55),
        "expert": NetworkProfile(DMN=0.15, VAN=0.5, DAN=0.8, FPN=0.7)
    }
}
```

**Literature References**:
- **DMN Suppression**: Brewer et al., 2022; Hasenkamp & Barsalou, 2012
- **DAN/FPN Enhancement**: Tang et al., 2015; Lutz et al., 2008
- **VAN Salience**: Seeley et al., 2007; Menon & Uddin, 2010
- **DMN-DAN Anticorrelation**: Fox et al., 2005; Spreng et al., 2013

---

## Active Inference Parameter Configuration

### TransitionThresholds Dataclass

```python
@dataclass
class TransitionThresholds:
    """Thresholds for state transitions"""
    mind_wandering: float      # Distraction level threshold
    dmn_dan_ratio: float      # DMN/DAN ratio threshold
    meta_awareness: float     # Self-reflection threshold for meta-awareness
    return_focus: float       # Threshold to return to focused states
```

#### Experience-Level Thresholds

```python
@classmethod
def novice(cls) -> 'TransitionThresholds':
    return cls(
        mind_wandering=0.6,    # Easier to get distracted
        dmn_dan_ratio=0.5,     # Lower threshold for DMN dominance
        meta_awareness=0.4,    # Higher threshold for awareness
        return_focus=0.3       # Higher threshold to return to focus
    )

@classmethod
def expert(cls) -> 'TransitionThresholds':
    return cls(
        mind_wandering=0.7,    # Harder to get distracted
        dmn_dan_ratio=0.6,     # Higher threshold for DMN dominance
        meta_awareness=0.3,    # Lower threshold for awareness (more sensitive)
        return_focus=0.25      # Lower threshold to return to focus (easier)
    )
```

### ActiveInferenceParameters Dataclass

```python
@dataclass
class ActiveInferenceParameters:
    """Core active inference parameters"""
    precision_weight: float
    complexity_penalty: float
    learning_rate: float
    noise_level: float
    memory_factor: float
    fpn_enhancement: float
    transition_thresholds: TransitionThresholds
    network_modulation: NetworkModulationConfig
```

#### Experience-Level Parameters

| Parameter | Novice | Expert | Interpretation |
|-----------|--------|--------|----------------|
| `precision_weight` | 0.4 | 0.5 | Meta-awareness influence on precision |
| `complexity_penalty` | 0.4 | 0.2 | Model complexity constraint |
| `learning_rate` | 0.01 | 0.02 | Parameter adaptation speed |
| `noise_level` | 0.06 | 0.03 | Neural variability |
| `memory_factor` | 0.7 | 0.85 | Temporal smoothing strength |
| `fpn_enhancement` | 1.0 | 1.2 | Executive network boost |

---

## Network Modulation Configuration

### NetworkModulationConfig Dataclass

```python
@dataclass
class NetworkModulationConfig:
    """Centralized configuration for all network modulation effects"""
    
    # Network enhancement/suppression effects
    dmn_boost: Dict[str, ContextualEffect] = field(default_factory=dict)
    dmn_suppression: Dict[str, ContextualEffect] = field(default_factory=dict)
    dan_boost: Dict[str, ContextualEffect] = field(default_factory=dict)
    dan_suppression: Dict[str, ContextualEffect] = field(default_factory=dict)
    van_boost: Dict[str, ContextualEffect] = field(default_factory=dict)
    fpn_boost: Dict[str, ContextualEffect] = field(default_factory=dict)
    
    # Anticorrelation mechanisms
    dmn_dan_anticorrelation: Optional[ContextualEffect] = field(default=None)
    dan_dmn_anticorrelation: Optional[ContextualEffect] = field(default=None)
    
    # Memory and temporal dynamics
    memory_factor: Optional[ContextualEffect] = field(default=None)
    
    # Non-linear dynamics
    non_linear_dynamics: NonLinearDynamicsConfig = field(default_factory=NonLinearDynamicsConfig)
```

### Novice Network Modulation

```python
@classmethod
def novice(cls) -> 'NetworkModulationConfig':
    config = cls()
    
    # DMN effects
    config.dmn_boost["mind_wandering"] = ContextualEffect(base_value=EffectMagnitude.WEAK)
    config.dmn_suppression["breath_control"] = ContextualEffect(base_value=EffectMagnitude.STRONG)
    config.dmn_suppression["meta_awareness"] = ContextualEffect(base_value=EffectMagnitude.MODERATE)
    
    # DAN effects
    config.dan_boost["breath_control"] = ContextualEffect(base_value=EffectMagnitude.MODERATE)
    config.dan_suppression["mind_wandering"] = ContextualEffect(base_value=EffectMagnitude.WEAK)
    
    # VAN effects
    config.van_boost["meta_awareness"] = ContextualEffect(base_value=EffectMagnitude.MODERATE)
    
    # FPN effects
    config.fpn_boost["meta_awareness"] = ContextualEffect(base_value=EffectMagnitude.MODERATE)
    
    # Anticorrelation (weaker for novices)
    config.dmn_dan_anticorrelation = ContextualEffect(base_value=EffectMagnitude.WEAK)
    config.dan_dmn_anticorrelation = ContextualEffect(base_value=EffectMagnitude.WEAK)
    
    return config
```

### Expert Network Modulation

```python
@classmethod
def expert(cls) -> 'NetworkModulationConfig':
    config = cls()
    
    # Enhanced DMN control
    config.dmn_boost["mind_wandering"] = ContextualEffect(base_value=EffectMagnitude.MODERATE)
    config.dmn_suppression["breath_control"] = ContextualEffect(base_value=EffectMagnitude.STRONG)
    config.dmn_suppression["redirect_breath"] = ContextualEffect(base_value=EffectMagnitude.STRONG)
    
    # Enhanced DAN control
    config.dan_boost["breath_control"] = ContextualEffect(base_value=EffectMagnitude.MODERATE)
    config.dan_suppression["mind_wandering"] = ContextualEffect(base_value=EffectMagnitude.STRONG)
    
    # Enhanced VAN and FPN
    config.van_boost["meta_awareness"] = ContextualEffect(base_value=EffectMagnitude.STRONG)
    config.fpn_boost["meta_awareness"] = ContextualEffect(base_value=EffectMagnitude.STRONG)
    
    # Stronger anticorrelation
    config.dmn_dan_anticorrelation = ContextualEffect(base_value=EffectMagnitude.WEAK)
    config.dan_dmn_anticorrelation = ContextualEffect(base_value=EffectMagnitude.STRONG)
    
    return config
```

---

## Non-Linear Dynamics Configuration

### NonLinearDynamicsConfig Dataclass

```python
@dataclass
class NonLinearDynamicsConfig:
    """Configuration for non-linear network dynamics"""
    # Compression thresholds
    high_compression_threshold: float = 0.8
    low_compression_threshold: float = 0.2
    
    # Compression factors
    high_compression_factor: float = EffectMagnitude.MODERATE  # 0.2
    low_compression_factor: float = EffectMagnitude.MODERATE   # 0.2
    
    # DMN-DAN anticorrelation boundary factors
    anticorrelation_boundary_threshold: float = 0.3
    
    # Extreme value prevention
    max_dmn_mind_wandering: float = 0.85
    
    # Rapid change detection
    rapid_change_threshold: float = 0.15
    rapid_change_smoothing: float = 0.7
```

**Compression Function**:
- Values above 0.8 are compressed by factor 0.2
- Values below 0.2 are compressed by factor 0.2
- Prevents extreme network activations while maintaining responsiveness

**Anticorrelation Boundaries**:
- Effects diminish when networks approach boundary values (0.3)
- Prevents oscillatory behavior at activation limits

---

## Thoughtseed Target Activation System

### StateTargetActivations Dataclass

```python
@dataclass
class StateTargetActivations:
    """Target activations for each thoughtseed in a particular state"""
    breath_focus: float
    equanimity: float
    pain_discomfort: float
    pending_tasks: float
    self_reflection: float
```

### ThoughtseedParams Configuration Class

```python
@dataclass
class ThoughtseedParams:
    """Thoughtseed target activation parameters derived from meditation literature."""
    
    BASE_ACTIVATIONS = {
        "breath_control": StateTargetActivations(
            breath_focus=0.7,      # High focus during breath control
            equanimity=0.3,        # Moderate equanimity
            pain_discomfort=0.15,  # Low distraction
            pending_tasks=0.1,     # Low distraction
            self_reflection=0.2    # Low-moderate introspection
        ),
        "mind_wandering": StateTargetActivations(
            breath_focus=0.1,      # Very low focus
            equanimity=0.1,        # Low equanimity
            pain_discomfort=0.6,   # High distraction
            pending_tasks=0.7,     # High distraction
            self_reflection=0.1    # Low introspection
        ),
        # ... additional states
    }
```

#### Meta-Awareness Modulation

```python
META_AWARENESS_MODULATORS = {
    "breath_control": {
        "breath_focus": 0.1,       # Meta-awareness enhances focus
        "equanimity": 0.2,         # Meta-awareness enhances equanimity
        "pain_discomfort": 0.0,    # No effect on pain
        "pending_tasks": 0.0,      # No effect on tasks
        "self_reflection": 0.1     # Slight enhancement of reflection
    },
    "mind_wandering": {
        "breath_focus": 0.0,       # No enhancement during wandering
        "equanimity": 0.0,         # No enhancement during wandering
        "pain_discomfort": -0.1,   # Meta-awareness reduces pain focus
        "pending_tasks": -0.1,     # Meta-awareness reduces task focus
        "self_reflection": 0.3     # Strong enhancement of reflection
    }
    # ... additional states
}
```

#### Expert Adjustments

```python
EXPERT_ADJUSTMENTS = {
    "breath_control": {
        "breath_focus": 0.1,       # Experts have 10% higher breath focus
        "equanimity": 0.15,        # Experts have 15% higher equanimity
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
    # ... additional states
}
```

---

## Meta-Cognition Parameter System

### MetacognitionParams Configuration Class

```python
@dataclass
class MetacognitionParams:
    """Meta-awareness parameters based on empirical studies of meditation."""
    
    BASE_AWARENESS = {
        "breath_control": 0.5,     # Moderate meta-awareness during focus
        "mind_wandering": 0.25,    # Low meta-awareness during wandering
        "meta_awareness": 0.7,     # High meta-awareness during detection
        "redirect_breath": 0.6     # Moderate-high during redirection
    }
    
    THOUGHTSEED_INFLUENCES = {
        "self_reflection": 0.2,    # Self-reflection strongly enhances awareness
        "equanimity": 0.15         # Equanimity moderately enhances awareness
    }
    
    EXPERIENCE_BOOST = {
        "novice": 0.0,             # No boost for novices
        "expert": 0.2              # 20% boost for experts
    }
    
    EXPERT_EFFICIENCY = {
        "meta_awareness": 0.8,     # 20% reduction in explicit awareness (more efficient)
        "other_states": 0.9        # 10% better background awareness
    }
```

### Meta-Awareness Calculation Method

```python
@staticmethod
def calculate_meta_awareness(state, thoughtseed_activations, experience_level='novice'):
    # Get base awareness for this state
    base_awareness = MetacognitionParams.BASE_AWARENESS[state]
    
    # Calculate thoughtseed influence
    awareness_boost = 0
    for ts, influence in MetacognitionParams.THOUGHTSEED_INFLUENCES.items():
        if ts in thoughtseed_activations:
            awareness_boost += thoughtseed_activations[ts] * influence
    
    # Add experience boost
    experience_boost = MetacognitionParams.EXPERIENCE_BOOST[experience_level]
    
    # Calculate total
    meta_awareness = base_awareness + awareness_boost + experience_boost
    
    # Apply expert efficiency adjustments
    if experience_level == 'expert':
        if state == "meta_awareness":
            meta_awareness *= MetacognitionParams.EXPERT_EFFICIENCY["meta_awareness"]
        else:
            meta_awareness = max(0.3, meta_awareness * MetacognitionParams.EXPERT_EFFICIENCY["other_states"])
    
    return meta_awareness
```

---

## Configuration Access System

### ActiveInferenceConfig Access Class

```python
class ActiveInferenceConfig:
    """
    Centralized configuration for active inference parameters.
    Access parameters with ActiveInferenceConfig.get_params(experience_level)
    """
    @staticmethod
    def get_params(experience_level):
        """Get all active inference parameters for the specified experience level"""
        if experience_level == 'expert':
            return ActiveInferenceParameters.expert().as_dict()
        else:
            return ActiveInferenceParameters.novice().as_dict()
```

### Parameter Dictionary Conversion

```python
def as_dict(self) -> Dict[str, Union[float, Dict]]:
    """Convert to dictionary for compatibility with existing code"""
    return {
        'precision_weight': self.precision_weight,
        'complexity_penalty': self.complexity_penalty,
        'learning_rate': self.learning_rate,
        'noise_level': self.noise_level,
        'memory_factor': self.memory_factor,
        'fpn_enhancement': self.fpn_enhancement,
        'transition_thresholds': {
            'mind_wandering': self.transition_thresholds.mind_wandering,
            'dmn_dan_ratio': self.transition_thresholds.dmn_dan_ratio,
            'meta_awareness': self.transition_thresholds.meta_awareness,
            'return_focus': self.transition_thresholds.return_focus
        },
        'network_modulation': self.network_modulation.__dict__
    }
```

---

## Configuration Usage Patterns

### Loading Parameters in ActInfLearner

```python
class ActInfLearner:
    def __init__(self, experience_level='novice', timesteps_per_cycle=200):
        # Get all active inference parameters from centralized config
        aif_params = ActiveInferenceConfig.get_params(experience_level)
        
        # Unpack parameters
        self.precision_weight = aif_params['precision_weight']
        self.complexity_penalty = aif_params['complexity_penalty']
        self.learning_rate = aif_params['learning_rate']
        self.noise_level = aif_params['noise_level']
        self.memory_factor = aif_params['memory_factor']
        self.fpn_enhancement = aif_params['fpn_enhancement']
        self.transition_thresholds = aif_params['transition_thresholds']
        self.network_modulation = aif_params['network_modulation']
```

### Using Contextual Effects

```python
# Get VAN boost effect for meta_awareness state
van_boost_effect = self.network_modulation.get('van_boost', {}).get('meta_awareness')
if van_boost_effect:
    van_boost = van_boost_effect.get_value(current_state, self.experience_level)
    network_acts['VAN'] += van_boost * meta_awareness
```

### Accessing Thoughtseed Parameters

```python
# Get target activations for current state
targets_dict = ThoughtseedParams.get_target_activations(
    state, meta_awareness, self.experience_level)

# Convert to numpy array
target_activations = np.array([targets_dict[ts] for ts in self.thoughtseeds])
```

---

## Validation and Type Safety

### Dataclass Benefits

1. **Type Safety**: Automatic type checking for all parameters
2. **Default Values**: Sensible defaults prevent missing parameter errors
3. **Immutability**: Dataclasses prevent accidental parameter modification
4. **Documentation**: Self-documenting code with clear parameter meanings

### Parameter Validation

```python
# Example validation in dataclass
@dataclass
class TransitionThresholds:
    mind_wandering: float
    
    def __post_init__(self):
        assert 0.0 <= self.mind_wandering <= 1.0, "Threshold must be in [0,1] range"
```

### Configuration Testing

The configuration system supports easy testing:

```python
def test_expert_novice_differences():
    expert_params = ActiveInferenceParameters.expert()
    novice_params = ActiveInferenceParameters.novice()
    
    assert expert_params.precision_weight > novice_params.precision_weight
    assert expert_params.complexity_penalty < novice_params.complexity_penalty
    assert expert_params.learning_rate > novice_params.learning_rate
```

---

## Future Enhancements

### Dynamic Configuration

Future implementations could include:

1. **Runtime Parameter Updates**: Dynamic configuration changes during simulation
2. **Configuration Profiles**: Multiple experience profiles beyond novice/expert
3. **Parameter Learning**: Automatic parameter optimization based on empirical data
4. **Configuration Validation**: Comprehensive parameter range checking

### Advanced Features

1. **Hierarchical Configuration**: Nested parameter structures for complex behaviors
2. **Configuration Inheritance**: Base configurations with specialized overrides
3. **Parameter Uncertainty**: Probability distributions for parameter values
4. **Configuration Serialization**: Save/load configuration states

---

*This documentation covers the comprehensive configuration system that provides type-safe, modular parameter management for the meditation simulation framework. The dataclass-based approach ensures maintainable, well-documented, and validated parameter handling across all system components.*
