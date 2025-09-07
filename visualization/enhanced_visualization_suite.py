"""
Enhanced Visualization Suite - Comprehensive Plotting System.

This module orchestrates all visualization components to create a complete
suite of advanced plots highlighting meditation dynamics and free energy calculations.
"""

import os
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt

from .dynamics_visualizer import DynamicsVisualizer
from .advanced_free_energy_visualizer import AdvancedFreeEnergyVisualizer  
from .statistical_dashboard import StatisticalDashboard
from .plotting import load_json_data, set_plot_style


class EnhancedVisualizationSuite:
    """Complete visualization suite for meditation simulation analysis."""
    
    def __init__(self, output_dir: str = "results_act_inf/plots"):
        """Initialize the enhanced visualization suite."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize specialized visualizers
        self.dynamics_viz = DynamicsVisualizer()
        self.free_energy_viz = AdvancedFreeEnergyVisualizer()
        self.stats_dashboard = StatisticalDashboard()
        
    def generate_all_enhanced_visualizations(self) -> bool:
        """Generate all enhanced visualizations from saved data."""
        try:
            print("\n🎨 Generating Enhanced Visualization Suite...")
            
            # Load data
            print("   Loading simulation data...")
            novice_data = load_json_data('novice')
            expert_data = load_json_data('expert')
            
            if not novice_data or not expert_data:
                print("   ⚠️  Could not load simulation data files")
                return False
            
            # Set consistent style
            set_plot_style()
            
            success_count = 0
            total_plots = 0
            
            # 1. Dynamic Analysis Plots
            print("   📊 Creating dynamic analysis visualizations...")
            
            plots_to_generate = [
                # Dynamics visualizations
                (self._generate_phase_portraits, "Phase Portraits"),
                (self._generate_network_flow_diagrams, "Network Flow Diagrams"),
                (self._generate_thoughtseed_competition, "Thoughtseed Competition"),
                (self._generate_state_transition_analysis, "State Transition Analysis"),
                (self._generate_comprehensive_dashboard, "Comprehensive Dashboard"),
                
                # Advanced free energy visualizations
                (self._generate_energy_decomposition, "Energy Decomposition"),
                (self._generate_energy_surface_plots, "3D Energy Surfaces"),
                (self._generate_energy_flow_diagrams, "Energy Flow Analysis"),
                (self._generate_variational_landscapes, "Variational Landscapes"),
                
                # Statistical analysis
                (self._generate_statistical_dashboard, "Statistical Dashboard"),
                (self._generate_distribution_analysis, "Distribution Analysis"),
            ]
            
            for plot_func, plot_name in plots_to_generate:
                total_plots += 1
                try:
                    print(f"     - Generating {plot_name}...")
                    plot_func(novice_data, expert_data)
                    success_count += 1
                    print(f"       ✅ {plot_name} completed")
                except Exception as e:
                    print(f"       ❌ {plot_name} failed: {str(e)}")
            
            print(f"\n   📈 Enhanced visualization complete:")
            print(f"     - Success: {success_count}/{total_plots} plots generated")
            print(f"     - Output directory: {self.output_dir}")
            
            return success_count > 0
            
        except Exception as e:
            print(f"   ❌ Error in enhanced visualization suite: {str(e)}")
            return False
    
    def _generate_phase_portraits(self, novice_data: Dict[str, Any], expert_data: Dict[str, Any]):
        """Generate phase portrait plots."""
        self.dynamics_viz.create_phase_portrait(novice_data)
        self.dynamics_viz.create_phase_portrait(expert_data)
    
    def _generate_network_flow_diagrams(self, novice_data: Dict[str, Any], expert_data: Dict[str, Any]):
        """Generate network flow diagrams."""
        self.dynamics_viz.create_network_flow_diagram(novice_data)
        self.dynamics_viz.create_network_flow_diagram(expert_data)
    
    def _generate_thoughtseed_competition(self, novice_data: Dict[str, Any], expert_data: Dict[str, Any]):
        """Generate thoughtseed competition plots."""
        self.dynamics_viz.create_thoughtseed_competition_plot(novice_data)
        self.dynamics_viz.create_thoughtseed_competition_plot(expert_data)
    
    def _generate_state_transition_analysis(self, novice_data: Dict[str, Any], expert_data: Dict[str, Any]):
        """Generate state transition analysis."""
        self.dynamics_viz.create_state_transition_analysis(novice_data)
        self.dynamics_viz.create_state_transition_analysis(expert_data)
    
    def _generate_comprehensive_dashboard(self, novice_data: Dict[str, Any], expert_data: Dict[str, Any]):
        """Generate comprehensive dashboard."""
        self.dynamics_viz.create_comprehensive_dashboard(novice_data, expert_data)
    
    def _generate_energy_decomposition(self, novice_data: Dict[str, Any], expert_data: Dict[str, Any]):
        """Generate energy decomposition plots."""
        self.free_energy_viz.create_energy_decomposition_plot(novice_data)
        self.free_energy_viz.create_energy_decomposition_plot(expert_data)
    
    def _generate_energy_surface_plots(self, novice_data: Dict[str, Any], expert_data: Dict[str, Any]):
        """Generate 3D energy surface plots."""
        self.free_energy_viz.create_energy_surface_plot(novice_data, expert_data)
    
    def _generate_energy_flow_diagrams(self, novice_data: Dict[str, Any], expert_data: Dict[str, Any]):
        """Generate energy flow diagrams."""
        self.free_energy_viz.create_energy_flow_diagram(novice_data)
        self.free_energy_viz.create_energy_flow_diagram(expert_data)
    
    def _generate_variational_landscapes(self, novice_data: Dict[str, Any], expert_data: Dict[str, Any]):
        """Generate variational landscape plots."""
        self.free_energy_viz.create_variational_landscape(novice_data)
        self.free_energy_viz.create_variational_landscape(expert_data)
    
    def _generate_statistical_dashboard(self, novice_data: Dict[str, Any], expert_data: Dict[str, Any]):
        """Generate comprehensive statistical dashboard."""
        self.stats_dashboard.create_comprehensive_statistical_report(novice_data, expert_data)
    
    def _generate_distribution_analysis(self, novice_data: Dict[str, Any], expert_data: Dict[str, Any]):
        """Generate distribution analysis plots."""
        self.stats_dashboard.create_distribution_analysis(novice_data)
        self.stats_dashboard.create_distribution_analysis(expert_data)
    
    def create_visualization_index(self) -> str:
        """Create an index of all generated visualizations."""
        index_content = """
# Enhanced Meditation Visualization Suite

This directory contains comprehensive visualizations of meditation simulation data,
highlighting dynamics and free energy calculations.

## Visualization Categories

### 1. Dynamic Analysis
- **Phase Portraits**: Free energy vs meta-awareness phase space dynamics
- **Network Flow Diagrams**: Neural network activation and connectivity analysis  
- **Thoughtseed Competition**: Competition dynamics between different thoughtseeds
- **State Transition Analysis**: Meditation state transition patterns and probabilities
- **Comprehensive Dashboard**: Multi-panel overview of all dynamics

### 2. Advanced Free Energy Analysis
- **Energy Decomposition**: Detailed breakdown of free energy components
- **3D Energy Surfaces**: Three-dimensional energy landscape visualization
- **Energy Flow Analysis**: Energy dissipation, accumulation, and phase transitions
- **Variational Landscapes**: Precision-complexity parameter space analysis

### 3. Statistical Analysis
- **Statistical Dashboard**: Comprehensive statistical comparison and hypothesis testing
- **Distribution Analysis**: Detailed statistical distributions with fitted models

## Generated Files

### Dynamics Visualizations
"""
        
        # List expected files
        files_info = [
            ("phase_portrait_novice.png", "Novice phase portrait"),
            ("phase_portrait_expert.png", "Expert phase portrait"),
            ("network_flow_novice.png", "Novice network dynamics"),
            ("network_flow_expert.png", "Expert network dynamics"),
            ("thoughtseed_competition_novice.png", "Novice thoughtseed analysis"),
            ("thoughtseed_competition_expert.png", "Expert thoughtseed analysis"),
            ("state_transitions_novice.png", "Novice state transitions"),
            ("state_transitions_expert.png", "Expert state transitions"),
            ("comprehensive_dashboard.png", "Complete comparative dashboard"),
            ("energy_decomposition_novice.png", "Novice energy analysis"),
            ("energy_decomposition_expert.png", "Expert energy analysis"),
            ("energy_surface_3d.png", "3D energy surface comparison"),
            ("energy_flow_novice.png", "Novice energy flow"),
            ("energy_flow_expert.png", "Expert energy flow"),
            ("variational_landscape_novice.png", "Novice variational analysis"),
            ("variational_landscape_expert.png", "Expert variational analysis"),
            ("statistical_dashboard.png", "Complete statistical analysis"),
            ("distribution_analysis_novice.png", "Novice distributions"),
            ("distribution_analysis_expert.png", "Expert distributions"),
        ]
        
        for filename, description in files_info:
            filepath = os.path.join(self.output_dir, filename)
            exists = "✅" if os.path.exists(filepath) else "❌"
            index_content += f"- {exists} `{filename}`: {description}\n"
        
        index_content += """
## Usage Notes

- All visualizations use consistent color schemes for cross-plot comparison
- High-resolution (300 DPI) outputs suitable for publication
- Modular design allows individual plot regeneration
- Statistical significance testing included where applicable

## Interpretation Guide

### Free Energy Plots
- Lower free energy indicates more efficient meditation
- Smoother energy trajectories suggest better regulation
- Phase portraits reveal attractor dynamics

### Network Analysis  
- DMN-DAN anticorrelation indicates focused attention
- Higher FPN activation suggests cognitive control
- VAN activation reflects salience detection

### Statistical Metrics
- Effect sizes (Cohen's d) quantify expert-novice differences
- Correlation matrices reveal system interdependencies
- Distribution analysis shows individual variability patterns
"""
        
        # Save index file
        index_path = os.path.join(self.output_dir, "README.md")
        with open(index_path, 'w') as f:
            f.write(index_content)
        
        return index_path


def generate_enhanced_plots() -> bool:
    """Main function to generate all enhanced visualizations."""
    suite = EnhancedVisualizationSuite()
    success = suite.generate_all_enhanced_visualizations()
    
    if success:
        index_path = suite.create_visualization_index()
        print(f"\n📋 Visualization index created: {index_path}")
        
    return success


# Integration with existing visualization system
def enhance_main_visualization():
    """Enhance the main visualization system with new plot types."""
    from . import plotting
    
    # Add enhanced plotting to the main generate_all_plots function
    original_generate_all_plots = plotting.generate_all_plots
    
    def enhanced_generate_all_plots():
        """Enhanced version that includes both original and new visualizations."""
        print("\n🎨 Generating Complete Visualization Suite...")
        
        # Generate original plots
        print("   📊 Creating standard visualizations...")
        original_success = original_generate_all_plots()
        
        # Generate enhanced plots
        print("   🚀 Creating enhanced visualizations...")
        enhanced_success = generate_enhanced_plots()
        
        total_success = original_success and enhanced_success
        
        if total_success:
            print("   ✅ Complete visualization suite generated successfully!")
        else:
            print("   ⚠️  Some visualizations could not be generated")
        
        return total_success
    
    # Replace the original function
    plotting.generate_all_plots = enhanced_generate_all_plots
    
    return enhanced_generate_all_plots
