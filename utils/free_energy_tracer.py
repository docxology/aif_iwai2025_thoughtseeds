"""
Enhanced Free Energy calculation tracer with detailed logging.

This module provides comprehensive tracing and logging of all Free Energy calculations
in the Active Inference framework, including variational free energy, expected free energy,
prediction errors, and precision weighting components.
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class FreeEnergySnapshot:
    """Snapshot of all free energy components at a single timestep."""
    timestep: int
    state: str
    
    # Core free energy components
    variational_free_energy: float
    expected_free_energy: float
    prediction_error: float
    precision_weight: float
    complexity_penalty: float
    
    # Detailed breakdowns
    network_predictions: Dict[str, float]
    network_observations: Dict[str, float]
    network_prediction_errors: Dict[str, float]
    thoughtseed_activations: Dict[str, float]
    thoughtseed_predictions: Dict[str, float]
    
    # Meta-awareness components
    meta_awareness: float
    attention_precision: float
    cognitive_load: float
    
    # State transition components
    transition_probability: float
    state_entropy: float
    
    # Optimization components
    gradient_magnitude: float
    learning_rate_effective: float
    
    timestamp: str


@dataclass
class FreeEnergyTrace:
    """Complete trace of free energy evolution."""
    experience_level: str
    simulation_duration: int
    snapshots: List[FreeEnergySnapshot]
    summary_statistics: Dict[str, Any]
    optimization_metrics: Dict[str, Any]


class FreeEnergyTracer:
    """
    Comprehensive tracer for Free Energy calculations.
    
    Provides detailed logging, tracing, and analysis of all free energy
    components throughout the Active Inference process.
    """
    
    def __init__(self, output_dir: str = "./free_energy_traces"):
        """Initialize the tracer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.snapshots = []
        self.detailed_log = []
        self.component_histories = {
            'variational_fe': [],
            'expected_fe': [],
            'prediction_error': [],
            'precision': [],
            'complexity': []
        }
    
    def trace_timestep(self, learner: Any, timestep: int) -> FreeEnergySnapshot:
        """Create detailed snapshot of current free energy state."""
        
        # Get current state information
        current_state = learner.state_history[timestep] if timestep < len(learner.state_history) else "unknown"
        
        # Calculate detailed free energy components
        fe_components = self._calculate_detailed_components(learner, timestep)
        
        # Create comprehensive snapshot
        snapshot = FreeEnergySnapshot(
            timestep=timestep,
            state=current_state,
            
            # Core components
            variational_free_energy=float(fe_components['vfe']),
            expected_free_energy=float(fe_components['efe']),
            prediction_error=float(fe_components['prediction_error']),
            precision_weight=float(fe_components['precision']),
            complexity_penalty=float(fe_components['complexity']),
            
            # Network details
            network_predictions=fe_components['network_predictions'],
            network_observations=fe_components['network_observations'],
            network_prediction_errors=fe_components['network_prediction_errors'],
            
            # Thoughtseed details
            thoughtseed_activations=fe_components['thoughtseed_activations'],
            thoughtseed_predictions=fe_components['thoughtseed_predictions'],
            
            # Meta-awareness
            meta_awareness=float(fe_components['meta_awareness']),
            attention_precision=float(fe_components['attention_precision']),
            cognitive_load=float(fe_components['cognitive_load']),
            
            # State transition
            transition_probability=float(fe_components['transition_prob']),
            state_entropy=float(fe_components['state_entropy']),
            
            # Optimization
            gradient_magnitude=float(fe_components['gradient_mag']),
            learning_rate_effective=float(fe_components['lr_effective']),
            
            timestamp=datetime.now().isoformat()
        )
        
        self.snapshots.append(snapshot)
        
        # Update component histories
        self.component_histories['variational_fe'].append(snapshot.variational_free_energy)
        self.component_histories['expected_fe'].append(snapshot.expected_free_energy)
        self.component_histories['prediction_error'].append(snapshot.prediction_error)
        self.component_histories['precision'].append(snapshot.precision_weight)
        self.component_histories['complexity'].append(snapshot.complexity_penalty)
        
        # Log detailed information
        self._log_detailed_calculation(snapshot, fe_components)
        
        return snapshot
    
    def _calculate_detailed_components(self, learner: Any, timestep: int) -> Dict[str, Any]:
        """Calculate detailed breakdown of all free energy components."""
        components = {}
        
        # Get current network state
        if hasattr(learner, 'network_activations_history') and timestep < len(learner.network_activations_history):
            current_networks = learner.network_activations_history[timestep]
            components['network_observations'] = {net: float(val) for net, val in current_networks.items()}
        else:
            components['network_observations'] = {}
        
        # Get predicted network state (from learned profiles)
        if hasattr(learner, 'learned_network_profiles'):
            current_state = learner.state_history[timestep] if timestep < len(learner.state_history) else 'breath_control'
            if current_state in learner.learned_network_profiles:
                predicted_networks = learner.learned_network_profiles[current_state]
                components['network_predictions'] = {net: float(val) for net, val in predicted_networks.items()}
            else:
                components['network_predictions'] = {}
        else:
            components['network_predictions'] = {}
        
        # Calculate network prediction errors
        network_pe = {}
        for net in components['network_observations']:
            obs = components['network_observations'][net]
            pred = components['network_predictions'].get(net, obs)
            network_pe[net] = float((obs - pred) ** 2)
        components['network_prediction_errors'] = network_pe
        
        # Get thoughtseed activations
        if timestep < len(learner.activations_history):
            ts_activations = learner.activations_history[timestep]
            components['thoughtseed_activations'] = {
                ts: float(ts_activations[i]) for i, ts in enumerate(learner.thoughtseeds)
            }
        else:
            components['thoughtseed_activations'] = {}
        
        # Calculate thoughtseed predictions (simplified)
        components['thoughtseed_predictions'] = {
            ts: float(np.mean([components['thoughtseed_activations'].get(ts, 0.5)])) 
            for ts in learner.thoughtseeds
        }
        
        # Core free energy calculations
        prediction_error = np.mean(list(network_pe.values())) if network_pe else 0.0
        precision_weight = getattr(learner, 'precision_weight', 0.5)
        complexity_penalty = getattr(learner, 'complexity_penalty', 0.3)
        
        # Variational Free Energy (Equation 2): F = E_q[ln q(s) - ln p(o,s)]
        entropy_q = -np.sum([act * np.log(act + 1e-8) for act in components['thoughtseed_activations'].values()])
        log_joint = -prediction_error  # Simplified log p(o,s)
        vfe = entropy_q - log_joint
        
        # Expected Free Energy (Equation 4): G = E_q[ln q(s_t+1) - ln p(o_t+1,s_t+1)]
        future_entropy = entropy_q  # Approximation
        expected_accuracy = -prediction_error * 0.9  # Discounted future accuracy
        efe = future_entropy - expected_accuracy
        
        components.update({
            'vfe': vfe,
            'efe': efe,
            'prediction_error': prediction_error,
            'precision': precision_weight,
            'complexity': complexity_penalty,
            
            # Meta-awareness components
            'meta_awareness': float(learner.meta_awareness_history[timestep]) if timestep < len(learner.meta_awareness_history) else 0.5,
            'attention_precision': precision_weight * (1 + 0.5 * components.get('meta_awareness', 0.5)),
            'cognitive_load': float(np.std(list(components['thoughtseed_activations'].values()))),
            
            # State transition components
            'transition_prob': self._calculate_transition_probability(learner, timestep),
            'state_entropy': self._calculate_state_entropy(learner, timestep),
            
            # Optimization components
            'gradient_mag': float(abs(prediction_error * precision_weight)),
            'lr_effective': float(getattr(learner, 'learning_rate', 0.01) * precision_weight)
        })
        
        return components
    
    def _calculate_transition_probability(self, learner: Any, timestep: int) -> float:
        """Calculate state transition probability."""
        if timestep == 0 or timestep >= len(learner.state_history):
            return 0.5
        
        # Simple transition probability based on state changes
        current_state = learner.state_history[timestep]
        prev_state = learner.state_history[timestep - 1]
        
        # Higher probability if state changed (indicating natural transition)
        if current_state != prev_state:
            return 0.8
        else:
            return 0.2
    
    def _calculate_state_entropy(self, learner: Any, timestep: int) -> float:
        """Calculate entropy of state distribution."""
        if timestep >= len(learner.activations_history):
            return 0.0
        
        # Use thoughtseed activations as state probabilities
        activations = np.array(learner.activations_history[timestep])
        # Normalize to probabilities
        probs = activations / (np.sum(activations) + 1e-8)
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        
        return float(entropy)
    
    def _log_detailed_calculation(self, snapshot: FreeEnergySnapshot, components: Dict[str, Any]) -> None:
        """Log detailed calculation steps."""
        log_entry = {
            'timestep': snapshot.timestep,
            'state': snapshot.state,
            'calculation_details': {
                'variational_fe_breakdown': {
                    'entropy_term': components.get('entropy_q', 0),
                    'log_joint_term': components.get('log_joint', 0),
                    'total_vfe': snapshot.variational_free_energy
                },
                'expected_fe_breakdown': {
                    'future_entropy': components.get('future_entropy', 0),
                    'expected_accuracy': components.get('expected_accuracy', 0),
                    'total_efe': snapshot.expected_free_energy
                },
                'precision_weighting': {
                    'base_precision': components['precision'],
                    'meta_awareness_boost': components['meta_awareness'],
                    'effective_precision': snapshot.attention_precision
                }
            },
            'timestamp': snapshot.timestamp
        }
        
        self.detailed_log.append(log_entry)
    
    def create_trace_summary(self, learner: Any) -> FreeEnergyTrace:
        """Create comprehensive trace summary."""
        
        # Calculate summary statistics
        vfe_history = self.component_histories['variational_fe']
        efe_history = self.component_histories['expected_fe']
        
        summary_stats = {
            'variational_fe_stats': {
                'mean': float(np.mean(vfe_history)) if vfe_history else 0,
                'std': float(np.std(vfe_history)) if vfe_history else 0,
                'min': float(np.min(vfe_history)) if vfe_history else 0,
                'max': float(np.max(vfe_history)) if vfe_history else 0,
                'final': float(vfe_history[-1]) if vfe_history else 0,
                'initial': float(vfe_history[0]) if vfe_history else 0
            },
            'expected_fe_stats': {
                'mean': float(np.mean(efe_history)) if efe_history else 0,
                'std': float(np.std(efe_history)) if efe_history else 0,
                'convergence_timestep': self._find_convergence_point(efe_history)
            },
            'optimization_efficiency': {
                'total_reduction': float(vfe_history[0] - vfe_history[-1]) if len(vfe_history) > 1 else 0,
                'percent_reduction': float((vfe_history[0] - vfe_history[-1]) / vfe_history[0] * 100) if len(vfe_history) > 1 and vfe_history[0] != 0 else 0,
                'convergence_rate': self._calculate_convergence_rate(vfe_history)
            }
        }
        
        # Calculate optimization metrics
        optimization_metrics = {
            'gradient_evolution': [s.gradient_magnitude for s in self.snapshots],
            'learning_adaptation': [s.learning_rate_effective for s in self.snapshots],
            'precision_evolution': [s.attention_precision for s in self.snapshots],
            'complexity_cost': [s.complexity_penalty for s in self.snapshots]
        }
        
        trace = FreeEnergyTrace(
            experience_level=learner.experience_level,
            simulation_duration=len(self.snapshots),
            snapshots=self.snapshots,
            summary_statistics=summary_stats,
            optimization_metrics=optimization_metrics
        )
        
        return trace
    
    def _find_convergence_point(self, values: List[float]) -> int:
        """Find timestep where values converge."""
        if len(values) < 10:
            return -1
        
        # Look for point where variance becomes stable
        window_size = max(5, len(values) // 10)
        
        for i in range(window_size, len(values) - window_size):
            before_var = np.var(values[i-window_size:i])
            after_var = np.var(values[i:i+window_size])
            
            if after_var < 0.1 * before_var:
                return i
        
        return -1
    
    def _calculate_convergence_rate(self, values: List[float]) -> float:
        """Calculate rate of convergence."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend in final 50% of data
        second_half = values[len(values)//2:]
        x = np.arange(len(second_half))
        
        if len(second_half) < 2:
            return 0.0
        
        slope = np.polyfit(x, second_half, 1)[0]
        return float(abs(slope))
    
    def save_trace(self, trace: FreeEnergyTrace, filename: str) -> str:
        """Save complete trace to file."""
        filepath = self.output_dir / f"{filename}_fe_trace.json"
        
        # Convert trace to serializable format
        trace_dict = {
            'experience_level': trace.experience_level,
            'simulation_duration': trace.simulation_duration,
            'snapshots': [asdict(snapshot) for snapshot in trace.snapshots],
            'summary_statistics': trace.summary_statistics,
            'optimization_metrics': trace.optimization_metrics,
            'detailed_log': self.detailed_log
        }
        
        with open(filepath, 'w') as f:
            json.dump(trace_dict, f, indent=2, default=str)
        
        print(f"✅ Free energy trace saved: {filepath}")
        return str(filepath)
    
    def save_component_histories(self, filename: str) -> str:
        """Save component evolution histories."""
        filepath = self.output_dir / f"{filename}_fe_components.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.component_histories, f, indent=2)
        
        return str(filepath)
