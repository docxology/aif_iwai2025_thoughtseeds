#!/usr/bin/env python3
"""
Enhanced Meditation Simulation with Comprehensive Free Energy Tracing

This script runs the complete meditation simulation with enhanced free energy tracing,
comprehensive logging, and advanced visualization capabilities.

Features:
- Detailed free energy component tracking
- Real-time visualization of all calculations
- Enhanced logging and data export
- Comparative analysis between novice and expert
- Advanced statistical analysis
"""

import numpy as np
import argparse
import sys
from pathlib import Path

# Import all components
try:
    from core import ActInfLearner, RuleBasedLearner
    from config import ActiveInferenceConfig
    from utils import (
        ensure_directories, FreeEnergyTracer, ExportManager, ExportConfig
    )
    from visualization import generate_all_plots, FreeEnergyVisualizer
    from analysis import StatisticalAnalyzer, ComparisonAnalyzer, MetricsCalculator
    print("✅ All module imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class EnhancedActInfLearner(ActInfLearner):
    """Enhanced ActInfLearner with comprehensive free energy tracing."""
    
    def __init__(self, experience_level='novice', timesteps_per_cycle=200):
        super().__init__(experience_level, timesteps_per_cycle)
        
        # Initialize free energy tracer
        self.fe_tracer = FreeEnergyTracer(
            output_dir=f"./free_energy_traces/{experience_level}"
        )
        
        # Enhanced tracking
        self.detailed_fe_history = []
        self.component_breakdowns = []
        self.optimization_trajectory = []
        
    def enhanced_free_energy_calculation(self, network_acts, current_state, 
                                       meta_awareness, timestep):
        """Calculate free energy with detailed component tracking."""
        
        # Standard free energy calculation
        free_energy, prediction_errors, total_prediction_error = \
            self.calculate_free_energy(network_acts, current_state, meta_awareness)
        
        # Create detailed snapshot
        snapshot = self.fe_tracer.trace_timestep(self, timestep)
        
        # Store additional analysis data
        self.detailed_fe_history.append({
            'timestep': timestep,
            'free_energy': free_energy,
            'prediction_errors': prediction_errors,
            'total_prediction_error': total_prediction_error,
            'network_activations': network_acts.copy(),
            'state': current_state,
            'meta_awareness': meta_awareness,
            'snapshot_id': len(self.fe_tracer.snapshots) - 1
        })
        
        return free_energy, prediction_errors, total_prediction_error
    
    def train_with_enhanced_tracking(self):
        """Enhanced training loop with comprehensive tracking."""
        print(f"\n🧠 Starting Enhanced {self.experience_level.upper()} Training...")
        print(f"   📊 Timesteps: {self.timesteps}")
        print(f"   🔬 Free Energy Tracing: Enabled")
        print(f"   📈 Component Analysis: Enabled")
        
        # Create output directories
        ensure_directories('./results_enhanced')
        ensure_directories('./free_energy_traces')
        
        # Initialize training variables
        activations = np.array([0.5] * self.num_thoughtseeds)
        current_state = "breath_control"
        current_state_index = 0
        current_dwell = 0
        dwell_limit = self.get_dwell_time(current_state)
        
        # State sequence
        state_sequence = ["breath_control", "mind_wandering", "meta_awareness", "redirect_breath"]
        
        print(f"   🎯 Initial State: {current_state}")
        print(f"   ⏱️  Initial Dwell Limit: {dwell_limit}")
        print(f"   🚀 Starting simulation...\n")
        
        # Enhanced training loop
        for t in range(self.timesteps):
            # Progress indicator
            if t % (self.timesteps // 10) == 0:
                progress = (t / self.timesteps) * 100
                print(f"   Progress: {progress:.1f}% (t={t})")
            
            # 1. Calculate meta-awareness
            meta_awareness = self.get_meta_awareness(current_state, activations)
            
            # 2. Get target activations
            target_activations = self.get_target_activations(current_state, meta_awareness)
            
            # 3. Smooth transitions
            blend_factor = 0.9
            activations = target_activations * blend_factor + activations * (1 - blend_factor)
            
            # 4. Compute network activations
            network_acts = self.compute_network_activations(
                activations, current_state, meta_awareness)
            
            # 5. Enhanced free energy calculation with tracing
            free_energy, prediction_errors, total_prediction_error = \
                self.enhanced_free_energy_calculation(
                    network_acts, current_state, meta_awareness, t)
            
            # 6. Update network profiles
            self.update_network_profiles(
                activations, network_acts, current_state, prediction_errors)
            
            # 7. Apply network modulation
            activations = self.network_modulated_activations(
                activations, network_acts, current_state)
            
            # 8. Record histories
            self.record_timestep_data(
                activations, network_acts, current_state, 
                meta_awareness, free_energy, prediction_errors, 
                total_prediction_error, t)
            
            # 9. Handle state transitions
            current_dwell += 1
            if current_dwell >= dwell_limit:
                # Natural transition logic with free energy influence
                transition_occurred, new_state = self.enhanced_transition_logic(
                    current_state, activations, network_acts, free_energy, t)
                
                if transition_occurred:
                    current_state = new_state
                    current_dwell = 0
                    dwell_limit = self.get_dwell_time(current_state)
                    
                    print(f"   🔄 t={t}: Transition to {current_state} "
                          f"(dwell: {dwell_limit}, FE: {free_energy:.3f})")
        
        print(f"\n✅ Enhanced {self.experience_level.upper()} training completed!")
        
        # Generate comprehensive trace summary
        trace_summary = self.fe_tracer.create_trace_summary(self)
        
        # Save enhanced outputs
        self.save_enhanced_outputs(trace_summary)
        
        return trace_summary
    
    def enhanced_transition_logic(self, current_state, activations, network_acts, 
                                free_energy, timestep):
        """Enhanced state transition logic with free energy influence."""
        
        # Base transition probability influenced by free energy
        base_prob = 0.8
        fe_influence = min(0.2, free_energy * 0.1)  # Higher FE increases transition prob
        experience_factor = 1.2 if self.experience_level == 'expert' else 0.8
        
        transition_prob = min(0.95, base_prob + fe_influence * experience_factor)
        
        if np.random.random() < transition_prob:
            # State-specific transition logic (simplified)
            if current_state == "breath_control":
                # Check for distraction
                distraction = (activations[self.thoughtseeds.index("pain_discomfort")] + 
                             activations[self.thoughtseeds.index("pending_tasks")])
                if distraction > 0.6:
                    return True, "mind_wandering"
                    
            elif current_state == "mind_wandering":
                # Check for meta-awareness
                if activations[self.thoughtseeds.index("self_reflection")] > 0.4:
                    return True, "meta_awareness"
                    
            elif current_state == "meta_awareness":
                # Return to focused state
                if (activations[self.thoughtseeds.index("breath_focus")] > 0.3 or
                    activations[self.thoughtseeds.index("equanimity")] > 0.3):
                    if (self.experience_level == 'expert' and 
                        activations[self.thoughtseeds.index("equanimity")] > 
                        activations[self.thoughtseeds.index("breath_focus")]):
                        return True, "redirect_breath"
                    else:
                        return True, "breath_control"
                        
            elif current_state == "redirect_breath":
                return True, "breath_control"
        
        return False, current_state
    
    def record_timestep_data(self, activations, network_acts, current_state, 
                           meta_awareness, free_energy, prediction_errors, 
                           total_prediction_error, timestep):
        """Record comprehensive timestep data."""
        
        # Standard recording
        self.activations_history.append(activations.copy())
        self.network_activations_history.append(network_acts.copy())
        self.state_history.append(current_state)
        self.meta_awareness_history.append(meta_awareness)
        self.free_energy_history.append(free_energy)
        self.prediction_error_history.append(total_prediction_error)
        
        # Enhanced recording
        dominant_ts = self.thoughtseeds[np.argmax(activations)]
        self.dominant_ts_history.append(dominant_ts)
        
        # Precision calculation
        precision = 0.5 + self.precision_weight * meta_awareness
        self.precision_history.append(precision)
    
    def save_enhanced_outputs(self, trace_summary):
        """Save all enhanced outputs."""
        
        # Save free energy trace
        trace_file = self.fe_tracer.save_trace(
            trace_summary, f"enhanced_{self.experience_level}")
        
        # Save component histories
        component_file = self.fe_tracer.save_component_histories(
            f"enhanced_{self.experience_level}")
        
        # Export enhanced data
        export_config = ExportConfig.comprehensive()
        export_manager = ExportManager(export_config)
        
        enhanced_exports = export_manager.export_learner(
            self, f"enhanced_{self.experience_level}")
        
        print(f"   📁 Free energy trace: {trace_file}")
        print(f"   📁 Component histories: {component_file}")
        print(f"   📁 Enhanced exports: {len(enhanced_exports)} files")


def run_enhanced_simulation(experience_level='both', timesteps=200, 
                          generate_visualizations=True):
    """Run enhanced simulation with comprehensive analysis."""
    
    print("🔬 Enhanced Meditation Simulation with Free Energy Tracing")
    print("=" * 60)
    
    results = {}
    
    # Determine which experience levels to run
    if experience_level == 'both':
        levels = ['novice', 'expert']
    else:
        levels = [experience_level]
    
    # Run simulations
    for level in levels:
        print(f"\n🎯 Running {level.upper()} simulation...")
        
        learner = EnhancedActInfLearner(
            experience_level=level, 
            timesteps_per_cycle=timesteps
        )
        
        trace_summary = learner.train_with_enhanced_tracking()
        results[level] = {
            'learner': learner,
            'trace_summary': trace_summary
        }
        
        print(f"✅ {level.upper()} simulation completed!")
        print(f"   🔥 Final FE: {trace_summary.snapshots[-1].variational_free_energy:.3f}")
        print(f"   📊 Snapshots: {len(trace_summary.snapshots)}")
        print(f"   🎛️ Optimization: {trace_summary.optimization_metrics['gradient_evolution'][-1]:.4f}")
    
    # Comparative analysis
    if len(results) == 2:
        print(f"\n📈 Running Comparative Analysis...")
        
        comparison_analyzer = ComparisonAnalyzer()
        comparison = comparison_analyzer.compare_learners(
            results['novice']['learner'], 
            results['expert']['learner']
        )
        
        # Enhanced visualization
        if generate_visualizations:
            print(f"\n🎨 Generating Enhanced Visualizations...")
            
            # Standard plots
            generate_all_plots()
            
            # Enhanced free energy visualizations
            fe_visualizer = FreeEnergyVisualizer(
                output_dir="./free_energy_visualizations"
            )
            
            # Create comprehensive dashboards
            novice_dashboard = fe_visualizer.create_comprehensive_dashboard(
                results['novice']['trace_summary']
            )
            
            expert_dashboard = fe_visualizer.create_comprehensive_dashboard(
                results['expert']['trace_summary']
            )
            
            comparative_dashboard = fe_visualizer.create_comparative_analysis(
                results['novice']['trace_summary'],
                results['expert']['trace_summary']
            )
            
            # Detailed component analysis
            novice_components = fe_visualizer.create_detailed_component_analysis(
                results['novice']['trace_summary']
            )
            
            expert_components = fe_visualizer.create_detailed_component_analysis(
                results['expert']['trace_summary']
            )
            
            # Optimization landscapes
            novice_landscape = fe_visualizer.create_optimization_landscape(
                results['novice']['trace_summary']
            )
            
            expert_landscape = fe_visualizer.create_optimization_landscape(
                results['expert']['trace_summary']
            )
            
            print(f"   📊 Dashboard files generated")
            print(f"   🔍 Component analysis files generated")
            print(f"   🗺️  Optimization landscapes generated")
    
    print(f"\n🎉 Enhanced simulation complete!")
    print(f"   📁 Results saved in: ./results_enhanced/")
    print(f"   🔬 Traces saved in: ./free_energy_traces/")
    print(f"   🎨 Visualizations saved in: ./free_energy_visualizations/")
    
    return results


def main():
    """Main entry point for enhanced simulation."""
    parser = argparse.ArgumentParser(
        description="Enhanced Meditation Simulation with Free Energy Tracing"
    )
    
    parser.add_argument(
        '--experience', 
        choices=['novice', 'expert', 'both'], 
        default='both',
        help='Experience level to simulate'
    )
    
    parser.add_argument(
        '--timesteps', 
        type=int, 
        default=200,
        help='Number of timesteps per simulation'
    )
    
    parser.add_argument(
        '--no-visualizations', 
        action='store_true',
        help='Skip visualization generation'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_enhanced_simulation(
            experience_level=args.experience,
            timesteps=args.timesteps,
            generate_visualizations=not args.no_visualizations
        )
        
        print(f"\n✨ All processes completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during enhanced simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
