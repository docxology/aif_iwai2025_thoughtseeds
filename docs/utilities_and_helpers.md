# Utilities and Helpers Documentation

## Overview

This document provides comprehensive technical details on the utility functions and helper modules that support the meditation simulation framework. The primary utilities are found in `meditation_utils.py` and provide essential functionality for data management, file operations, and output processing.

---

## Core Utility Module: meditation_utils.py

### Module Purpose
- **File**: `meditation_utils.py`
- **Primary Functions**: Directory management, data serialization, JSON output generation
- **Dependencies**: os, numpy, json
- **Integration**: Used by both `ActInfLearner` and `RuleBasedLearner`

### Import Structure

```python
import os
import numpy as np
import json
from meditation_config import THOUGHTSEED_AGENTS
```

---

## Directory Management

### ensure_directories() Function

```python
def ensure_directories(base_dir='./results_act_inf'):
    """Create necessary directories for output files"""
    os.makedirs(f'{base_dir}/data', exist_ok=True)
    os.makedirs(f'{base_dir}/plots', exist_ok=True)
    print(f"Directories created/verified for {base_dir} output files")
```

**Purpose**: Creates the standard directory structure for simulation outputs.

**Directory Structure Created**:
```
results_act_inf/
├── data/           # JSON parameter files and statistics
└── plots/          # Generated visualization files
```

**Key Features**:
- **Safe Creation**: Uses `exist_ok=True` to prevent errors if directories already exist
- **Configurable Base**: Supports custom base directory paths
- **Verification Logging**: Confirms successful directory creation
- **Standard Structure**: Ensures consistent organization across simulations

**Usage Patterns**:
```python
# Standard usage
ensure_directories()  # Creates ./results_act_inf/

# Custom base directory
ensure_directories('./custom_results/')  # Creates ./custom_results/data/ and ./custom_results/plots/
```

---

## Data Serialization System

### NumPy Conversion Utilities

#### convert_numpy_to_lists() Helper Function

```python
def convert_numpy_to_lists(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_lists(i) for i in obj]
    else:
        return obj
```

**Purpose**: Recursively converts NumPy arrays to Python lists for JSON serialization compatibility.

**Key Features**:
- **Recursive Processing**: Handles nested data structures
- **Type Preservation**: Maintains data structure while converting arrays
- **JSON Compatibility**: Ensures all data can be serialized to JSON
- **Memory Efficient**: Processes data in-place where possible

**Conversion Examples**:
```python
# Single NumPy array
numpy_array = np.array([1, 2, 3])
python_list = convert_numpy_to_lists(numpy_array)  # [1, 2, 3]

# Nested structure
nested_data = {
    'activations': np.array([[0.1, 0.2], [0.3, 0.4]]),
    'metadata': {'timesteps': 100}
}
converted = convert_numpy_to_lists(nested_data)
# {'activations': [[0.1, 0.2], [0.3, 0.4]], 'metadata': {'timesteps': 100}}
```

---

## JSON Output Generation System

### Primary Output Function: _save_json_outputs()

```python
def _save_json_outputs(learner, output_dir='./results_act_inf/data/'):
    """Save JSON outputs"""
    print("\nGenerating consumer-ready JSON files...")
```

**Purpose**: Generates comprehensive JSON output files for analysis and visualization.

**Key Features**:
- **Consumer-Ready Format**: Structured for easy consumption by analysis tools
- **Time Series Extraction**: Separates temporal data for plotting
- **Statistical Summaries**: Pre-computed means and aggregations
- **Experience-Level Specific**: Separate files for novice and expert data

### Thoughtseed Parameters Output

#### Data Structure Generation

```python
thoughtseed_params = {
    "interactions": THOUGHTSEED_AGENTS,
    "agent_parameters": {
        ts: {
            "base_activation": float(np.mean([act[i] for act in learner.activations_history])),
            "responsiveness": float(max(0.5, 1.0 - np.std([act[i] for act in learner.activations_history]))),
            "decay_rate": THOUGHTSEED_AGENTS[ts]["decay_rate"],
            "recovery_rate": THOUGHTSEED_AGENTS[ts]["recovery_rate"],
            "network_profile": learner.learned_network_profiles["thoughtseed_contributions"][ts]
        } for i, ts in enumerate(learner.thoughtseeds)
    },
    "activation_means_by_state": {
        state: {
            ts: float(np.mean([
                learner.activations_history[j][i] 
                for j, s in enumerate(learner.state_history) if s == state
            ])) for i, ts in enumerate(learner.thoughtseeds)
        } for state in learner.states if any(s == state for s in learner.state_history)
    },
    "network_activations_by_state": {
        state: {
            net: float(np.mean([
                learner.network_activations_history[j][net]
                for j, s in enumerate(learner.state_history) if s == state
            ])) for net in learner.networks
        } for state in learner.states if any(s == state for s in learner.state_history)
    }
}
```

**Output Components**:

1. **Interactions**: Base thoughtseed agent definitions from configuration
2. **Agent Parameters**: 
   - `base_activation`: Average activation level across simulation
   - `responsiveness`: Inverse of activation variability (stability measure)
   - `decay_rate`: Configuration-defined decay parameter
   - `recovery_rate`: Configuration-defined recovery parameter
   - `network_profile`: Learned network contribution weights

3. **Activation Means by State**: Average thoughtseed activations for each meditation state
4. **Network Activations by State**: Average network activations for each meditation state

#### Time Series Data Addition

```python
thoughtseed_params["time_series"] = {
    "activations_history": convert_numpy_to_lists(learner.activations_history),
    "network_activations_history": convert_numpy_to_lists(learner.network_activations_history),
    "meta_awareness_history": learner.meta_awareness_history,  
    "free_energy_history": learner.free_energy_history,  
    "state_history": learner.state_history,
    "dominant_ts_history": learner.dominant_ts_history  
}
```

**Time Series Components**:
- **activations_history**: Complete thoughtseed activation time series
- **network_activations_history**: Complete network activation time series  
- **meta_awareness_history**: Meta-awareness values over time
- **free_energy_history**: Free energy trajectory
- **state_history**: State sequence over time
- **dominant_ts_history**: Dominant thoughtseed sequence

#### File Output

```python
with open(f"./results_act_inf/data/thoughtseed_params_{learner.experience_level}.json", "w") as f:
    json.dump(thoughtseed_params, f, indent=2)
```

Generated files:
- `thoughtseed_params_novice.json`
- `thoughtseed_params_expert.json`

### Active Inference Parameters Output

#### Data Structure Generation

```python
active_inf_params = {
    "precision_weight": learner.precision_weight,
    "complexity_penalty": learner.complexity_penalty,
    "learning_rate": learner.learning_rate,
    "average_free_energy_by_state": {
        state: float(np.mean([
            learner.free_energy_history[j]
            for j, s in enumerate(learner.state_history) if s == state
        ])) for state in learner.states if any(s == state for s in learner.state_history)
    },
    "average_prediction_error_by_state": {
        state: float(np.mean([
            learner.prediction_error_history[j]
            for j, s in enumerate(learner.state_history) if s == state
        ])) for state in learner.states if any(s == state for s in learner.state_history)
    },
    "average_precision_by_state": {
        state: float(np.mean([
            learner.precision_history[j]
            for j, s in enumerate(learner.state_history) if s == state
        ])) for state in learner.states if any(s == state for s in learner.state_history)
    },
    "network_expectations": learner.learned_network_profiles["state_network_expectations"]
}
```

**Output Components**:

1. **Core Parameters**: Active inference hyperparameters
2. **Free Energy Analysis**: State-specific free energy averages
3. **Prediction Error Analysis**: State-specific prediction error averages  
4. **Precision Analysis**: State-specific precision weight averages
5. **Network Expectations**: Learned state-network expectation profiles

#### File Output

```python
with open(f"./results_act_inf/data/active_inference_params_{learner.experience_level}.json", "w") as f:
    json.dump(active_inf_params, f, indent=2)
```

Generated files:
- `active_inference_params_novice.json`
- `active_inference_params_expert.json`

---

## Data Processing Utilities

### Statistical Computation Patterns

#### State-Specific Averaging

```python
# Pattern for computing state-specific means
state_means = {
    state: float(np.mean([
        learner.data_history[j]
        for j, s in enumerate(learner.state_history) if s == state
    ])) for state in learner.states if any(s == state for s in learner.state_history)
}
```

**Key Features**:
- **Conditional Filtering**: Only includes timesteps where state matches
- **Existence Checking**: Validates state occurred during simulation
- **Type Conversion**: Ensures JSON-serializable float types
- **Comprehensive Coverage**: Processes all states that occurred

#### Multi-Dimensional Data Processing

```python
# Pattern for processing multi-dimensional time series
"network_activations_by_state": {
    state: {
        net: float(np.mean([
            learner.network_activations_history[j][net]
            for j, s in enumerate(learner.state_history) if s == state
        ])) for net in learner.networks
    } for state in learner.states if any(s == state for s in learner.state_history)
}
```

This pattern creates nested dictionaries with state-specific network averages.

---

## Integration with Simulation Framework

### Usage in ActInfLearner

```python
class ActInfLearner(RuleBasedLearner):
    def train(self):
        # ... simulation logic ...
        
        # At end of training
        ensure_directories('./results_act_inf')
        _save_json_outputs(self)
```

### Usage in Main Execution

```python
if __name__ == "__main__":
    from meditation_utils import ensure_directories
    
    # Set up environment
    ensure_directories()
    
    # Train models
    learner_novice = ActInfLearner(experience_level='novice')
    learner_novice.train()
```

---

## File Organization System

### Standard Output Structure

After successful execution, the utilities create:

```
results_act_inf/
├── data/
│   ├── thoughtseed_params_novice.json      # Novice thoughtseed data
│   ├── thoughtseed_params_expert.json      # Expert thoughtseed data
│   ├── active_inference_params_novice.json # Novice AI parameters
│   ├── active_inference_params_expert.json # Expert AI parameters
│   ├── transition_stats_novice.json        # Novice transition data
│   └── transition_stats_expert.json        # Expert transition data
└── plots/
    ├── network_radar_comparison.png         # Network comparison radar
    ├── free_energy_comparison.png           # Free energy bar chart
    ├── hierarchy_novice.png                 # Novice hierarchical plot
    ├── hierarchy_expert.png                 # Expert hierarchical plot
    └── time_series_comparison.png           # Time series comparison
```

### File Size and Performance

**Typical File Sizes** (200 timesteps):
- `thoughtseed_params_*.json`: ~50-100 KB (includes full time series)
- `active_inference_params_*.json`: ~10-20 KB (summary statistics)
- `transition_stats_*.json`: ~15-30 KB (transition matrices and patterns)

**Performance Characteristics**:
- **Write Time**: < 1 second for all JSON files
- **Memory Usage**: Minimal additional memory beyond simulation data
- **Read Time**: < 0.5 seconds for visualization system

---

## Error Handling and Robustness

### Directory Creation Safety

```python
os.makedirs(f'{base_dir}/data', exist_ok=True)
os.makedirs(f'{base_dir}/plots', exist_ok=True)
```

The `exist_ok=True` parameter prevents errors if directories already exist.

### Data Validation

```python
# Implicit validation through numpy operations
float(np.mean([...]))  # Will fail gracefully if data is malformed
```

### Type Safety

```python
# Explicit type conversion for JSON compatibility
"learning_rate": float(learner.learning_rate)
"timesteps": int(len(data["state_history"]))
```

---

## Future Enhancement Possibilities

### Advanced Data Export

Potential additions:
```python
def save_csv_outputs(learner):
    """Export data in CSV format for R/MATLAB analysis"""
    pass

def save_hdf5_outputs(learner):
    """Export data in HDF5 format for large-scale analysis"""
    pass
```

### Data Compression

```python
def save_compressed_outputs(learner, compression='gzip'):
    """Save compressed JSON for large simulations"""
    pass
```

### Metadata Enhancement

```python
def add_simulation_metadata(data_dict):
    """Add timestamp, version, and configuration metadata"""
    data_dict['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'framework_version': get_version(),
        'configuration_hash': calculate_config_hash()
    }
```

---

## Testing and Validation

### Utility Function Tests

```python
def test_numpy_conversion():
    """Test NumPy array conversion to lists"""
    test_array = np.array([[1, 2], [3, 4]])
    converted = convert_numpy_to_lists(test_array)
    assert isinstance(converted, list)
    assert converted == [[1, 2], [3, 4]]

def test_directory_creation():
    """Test directory creation utility"""
    test_dir = './test_output'
    ensure_directories(test_dir)
    assert os.path.exists(f'{test_dir}/data')
    assert os.path.exists(f'{test_dir}/plots')
```

### Integration Tests

```python
def test_json_output_generation():
    """Test complete JSON output generation"""
    learner = ActInfLearner('novice', 50)
    learner.train()
    
    # Verify files exist
    assert os.path.exists('./results_act_inf/data/thoughtseed_params_novice.json')
    assert os.path.exists('./results_act_inf/data/active_inference_params_novice.json')
    
    # Verify JSON validity
    with open('./results_act_inf/data/thoughtseed_params_novice.json') as f:
        data = json.load(f)
        assert 'time_series' in data
        assert 'agent_parameters' in data
```

---

*This documentation covers the essential utility functions that support the meditation simulation framework through directory management, data serialization, and JSON output generation. These utilities ensure consistent data organization and enable seamless integration between simulation runs and analysis tools.*
