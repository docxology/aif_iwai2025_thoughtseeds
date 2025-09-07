#!/usr/bin/env python3
"""
Main execution script for the Dynamic Attentional Agents meditation simulation.

This script demonstrates the usage of the refactored modular framework while
preserving all original functionality. It provides a professional interface
for running simulations and generating comprehensive analyses with advanced visualizations.

Usage:
    python main.py [--experience novice|expert] [--timesteps N] [--plots] [--enhanced-plots] [--all-plots] [--validate]

Visualization Options:
    --plots            Generate standard visualization plots (radar, hierarchy, time series)
    --enhanced-plots   Generate advanced dynamics and free energy visualizations 
    --all-plots        Generate both standard and enhanced visualization suites
    
Enhanced visualizations include:
    • Phase portraits and dynamics analysis
    • Advanced free energy decomposition and landscapes
    • Statistical dashboards with hypothesis testing
    • Network flow and connectivity analysis
    • Thoughtseed competition dynamics
    • 3D energy surfaces and optimization trajectories
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Import from the new modular structure
from core import ActInfLearner, RuleBasedLearner
from config import ActiveInferenceConfig, THOUGHTSEEDS, STATES
from visualization import generate_all_plots, generate_enhanced_plots
from utils import ensure_directories, validate_data_integrity, get_latest_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Dynamic Attentional Agents in Focused Attention Meditation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        help='Number of timesteps per simulation cycle'
    )
    
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate standard visualization plots after simulation'
    )
    
    parser.add_argument(
        '--enhanced-plots',
        action='store_true',
        help='Generate enhanced visualization suite with advanced dynamics and free energy plots'
    )
    
    parser.add_argument(
        '--all-plots',
        action='store_true',
        help='Generate both standard and enhanced visualization suites'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true', 
        help='Validate data integrity after simulation'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible results'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results_act_inf',
        help='Directory for simulation outputs'
    )
    
    return parser.parse_args()


def run_simulation(experience_level: str, timesteps: int, output_dir: str) -> ActInfLearner:
    """
    Run a complete meditation simulation for the specified experience level.
    
    Args:
        experience_level: 'novice' or 'expert'
        timesteps: Number of simulation timesteps
        output_dir: Output directory for results
        
    Returns:
        Trained ActInfLearner instance
    """
    print(f"\n🧘 Starting {experience_level} meditation simulation...")
    print(f"   Timesteps: {timesteps}")
    print(f"   Output: {output_dir}")
    
    # Create learner instance
    learner = ActInfLearner(experience_level=experience_level, timesteps_per_cycle=timesteps)
    
    # Display configuration
    config = ActiveInferenceConfig.get_params(experience_level)
    print(f"   Configuration:")
    print(f"     - Precision weight: {config['precision_weight']}")
    print(f"     - Complexity penalty: {config['complexity_penalty']}")
    print(f"     - Learning rate: {config['learning_rate']}")
    print(f"     - Noise level: {config['noise_level']}")
    
    # Run simulation
    print(f"   Running simulation...")
    learner.train()
    
    print(f"✅ Completed {experience_level} simulation")
    print(f"   - Natural transitions: {learner.natural_transition_count}")
    print(f"   - Forced transitions: {learner.forced_transition_count}")
    
    return learner


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Create output directories
    print("🔧 Setting up output directories...")
    ensure_directories(args.output_dir)
    
    print(f"\n🎯 Dynamic Attentional Agents Meditation Simulation")
    print(f"   Framework: Active Inference with Hierarchical Thoughtseeds")
    print(f"   Architecture: {len(THOUGHTSEEDS)} thoughtseeds, {len(STATES)} states")
    print(f"   Random seed: {args.seed}")
    
    # Run simulations based on experience level selection
    results = {}
    
    if args.experience in ['novice', 'both']:
        results['novice'] = run_simulation('novice', args.timesteps, args.output_dir)
    
    if args.experience in ['expert', 'both']:
        results['expert'] = run_simulation('expert', args.timesteps, args.output_dir)
    
    # Generate comparative analysis
    if len(results) > 1:
        print(f"\n📊 Comparative Analysis:")
        novice_fe = np.mean(results['novice'].free_energy_history)
        expert_fe = np.mean(results['expert'].free_energy_history) 
        improvement = (novice_fe - expert_fe) / novice_fe * 100
        
        print(f"   Average Free Energy:")
        print(f"     - Novice: {novice_fe:.3f}")
        print(f"     - Expert: {expert_fe:.3f}")
        print(f"     - Expert improvement: {improvement:.1f}%")
    
    # Generate visualizations if requested
    plot_success = True
    
    if args.plots or args.all_plots:
        print(f"\n🎨 Generating standard visualization plots...")
        try:
            success = generate_all_plots()
            if success:
                print("✅ Standard visualization plots generated successfully")
            else:
                print("⚠️  Some standard plots could not be generated")
                plot_success = False
        except Exception as e:
            print(f"❌ Error generating standard plots: {e}")
            plot_success = False
    
    if args.enhanced_plots or args.all_plots:
        print(f"\n🚀 Generating enhanced visualization suite...")
        try:
            enhanced_success = generate_enhanced_plots()
            if enhanced_success:
                print("✅ Enhanced visualization suite generated successfully")
                print("   📊 Generated advanced dynamics visualizations")
                print("   ⚡ Generated detailed free energy analysis")
                print("   📈 Generated comprehensive statistical dashboard")
            else:
                print("⚠️  Some enhanced plots could not be generated")
                plot_success = False
        except Exception as e:
            print(f"❌ Error generating enhanced plots: {e}")
            plot_success = False
    
    if args.plots or args.enhanced_plots or args.all_plots:
        if plot_success:
            print(f"\n🎉 All requested visualizations completed successfully!")
        else:
            print(f"\n⚠️  Visualization generation completed with some issues")
    
    # Validate data integrity if requested
    if args.validate:
        print(f"\n🔍 Validating data integrity...")
        validation_results = validate_data_integrity(f"{args.output_dir}/data")
        
        all_valid = all(validation_results.values())
        if all_valid:
            print("✅ All data files validated successfully")
        else:
            print("⚠️  Data validation issues found:")
            for filename, valid in validation_results.items():
                status = "✅" if valid else "❌"
                print(f"     {status} {filename}")
    
    # Display results summary
    print(f"\n📈 Results Summary:")
    metadata = get_latest_results(args.output_dir)
    print(f"   Data files: {len(metadata['data_files'])}")
    print(f"   Plot files: {len(metadata['plot_files'])}")
    print(f"   Total size: {metadata['total_size'] / 1024:.1f} KB")
    print(f"   Output directory: {Path(args.output_dir).absolute()}")
    
    print(f"\n🎉 Simulation completed successfully!")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n⚠️  Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
