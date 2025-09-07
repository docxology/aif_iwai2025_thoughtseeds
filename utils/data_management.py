"""
Data management utilities for the meditation simulation framework.

This module contains all data management functions exactly as preserved from
the original meditation_utils.py implementation. No methods are changed.
"""

import os
import numpy as np
import json
from config import THOUGHTSEED_AGENTS


def ensure_directories(base_dir='./results_act_inf'):
    """Create necessary directories for output files"""
    os.makedirs(f'{base_dir}/data', exist_ok=True)
    os.makedirs(f'{base_dir}/plots', exist_ok=True)
    print(f"Directories created/verified for {base_dir} output files")


def convert_numpy_to_lists(obj):
    """Helper function to convert numpy arrays to lists for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_lists(i) for i in obj]
    else:
        return obj


def _save_json_outputs(learner, output_dir='./results_act_inf/data/'):
    """Save JSON outputs - exact preservation of original function"""
    print("\nGenerating consumer-ready JSON files...")
    
    # 1. ThoughtseedNetwork parameters with network profiles
    thoughtseed_params = {
        "interactions": {k: v.__dict__ for k, v in THOUGHTSEED_AGENTS.items()},
        "agent_parameters": {
            ts: {
                "base_activation": float(np.mean([act[i] for act in learner.activations_history])),
                "responsiveness": float(max(0.5, 1.0 - np.std([act[i] for act in learner.activations_history]))),
                "decay_rate": THOUGHTSEED_AGENTS[ts].decay_rate,
                "recovery_rate": THOUGHTSEED_AGENTS[ts].recovery_rate,
                "network_profile": learner.learned_network_profiles["thoughtseed_contributions"][ts].__dict__ 
                    if hasattr(learner.learned_network_profiles["thoughtseed_contributions"][ts], '__dict__') 
                    else dict(learner.learned_network_profiles["thoughtseed_contributions"][ts]) 
                    if hasattr(learner.learned_network_profiles["thoughtseed_contributions"][ts], 'keys')
                    else str(learner.learned_network_profiles["thoughtseed_contributions"][ts])
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
    
    # Add time series data (converting NumPy arrays to lists)
    thoughtseed_params["time_series"] = {
        "activations_history": convert_numpy_to_lists(learner.activations_history),
        "network_activations_history": convert_numpy_to_lists(learner.network_activations_history),
        "meta_awareness_history": learner.meta_awareness_history,  
        "free_energy_history": learner.free_energy_history,  
        "prediction_error_history": getattr(learner, 'prediction_error_history', []),
        "precision_history": getattr(learner, 'precision_history', []),
        "state_history": learner.state_history,
        "dominant_ts_history": learner.dominant_ts_history  
    }
    
    with open(f"./results_act_inf/data/thoughtseed_params_{learner.experience_level}.json", "w") as f:
        json.dump(thoughtseed_params, f, indent=2)
    
    # 2. Active Inference parameters
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
            "network_expectations": {
                state: {
                    level: expectations.__dict__ 
                        if hasattr(expectations, '__dict__') 
                        else dict(expectations) 
                        if hasattr(expectations, 'keys') 
                        else str(expectations)
                    for level, expectations in state_data.items()
                }
                for state, state_data in learner.learned_network_profiles["state_network_expectations"].items()
            }
    }
    
    with open(f"./results_act_inf/data/active_inference_params_{learner.experience_level}.json", "w") as f:
        json.dump(active_inf_params, f, indent=2)
    
    print(f"  - JSON parameter files saved to ./results_act_inf/data/ directory")
