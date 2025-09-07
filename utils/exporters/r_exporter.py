"""
R export functionality for statistical analysis integration.

This module provides R-compatible export capabilities for seamless
integration with R-based statistical analysis workflows. Exports data
in formats optimized for R's data.frame structure and analysis functions.
"""

import csv
from typing import Dict, Any, Optional, List
import numpy as np
from .base_exporter import BaseExporter


class RExporter(BaseExporter):
    """
    R-compatible data exporter for statistical analysis.
    
    Exports simulation data in R-friendly formats including CSV files
    optimized for R data.frame structure and generates R analysis scripts
    for common statistical tasks.
    """
    
    def __init__(self, output_dir: str = "./exports",
                 generate_script: bool = True,
                 long_format: bool = True,
                 **kwargs):
        """
        Initialize R exporter.
        
        Args:
            output_dir: Directory for exported files
            generate_script: Whether to generate R analysis script
            long_format: Export in long format (suitable for ggplot2)
            **kwargs: Additional arguments for base exporter
        """
        super().__init__(output_dir, **kwargs)
        self.generate_script = generate_script
        self.long_format = long_format
    
    def export(self, learner: Any, filename: Optional[str] = None) -> Dict[str, str]:
        """Export learner data in R-compatible format."""
        base_name = filename or "meditation_data"
        result = {}
        
        # Export main time series data
        if self.long_format:
            main_filepath = self._export_long_format(learner, f"{base_name}_long")
        else:
            main_filepath = self._export_wide_format(learner, f"{base_name}_wide")
        
        result['main_data'] = main_filepath
        
        # Export summary statistics
        summary_filepath = self._export_summary_data(learner, f"{base_name}_summary")
        result['summary_data'] = summary_filepath
        
        # Export network data if available
        if hasattr(learner, 'network_activations_history'):
            network_filepath = self._export_network_data(learner, f"{base_name}_networks")
            result['network_data'] = network_filepath
        
        # Generate R analysis script
        if self.generate_script:
            script_filepath = self._generate_r_script(learner, base_name, result)
            result['r_script'] = script_filepath
        
        return result
    
    def _export_long_format(self, learner: Any, base_name: str) -> str:
        """Export data in long format (tidy data principle)."""
        filepath = self._generate_filename(base_name, "csv", learner.experience_level)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Headers for long format
            headers = [
                'timestep', 'experience_level', 'state', 'meta_awareness',
                'dominant_thoughtseed', 'variable', 'value', 'variable_type'
            ]
            writer.writerow(headers)
            
            # Write thoughtseed activation data
            for t in range(len(learner.activations_history)):
                base_row = [
                    t,
                    learner.experience_level,
                    learner.state_history[t],
                    learner.meta_awareness_history[t],
                    learner.dominant_ts_history[t]
                ]
                
                # Thoughtseed activations
                for i, ts in enumerate(learner.thoughtseeds):
                    row = base_row + [ts, learner.activations_history[t][i], 'thoughtseed']
                    writer.writerow(row)
                
                # Network activations if available
                if hasattr(learner, 'network_activations_history'):
                    for net in learner.networks:
                        row = base_row + [net, learner.network_activations_history[t][net], 'network']
                        writer.writerow(row)
                    
                    # Free energy
                    row = base_row + ['free_energy', learner.free_energy_history[t], 'metric']
                    writer.writerow(row)
        
        return filepath
    
    def _export_wide_format(self, learner: Any, base_name: str) -> str:
        """Export data in wide format (one row per timestep)."""
        filepath = self._generate_filename(base_name, "csv", learner.experience_level)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Headers for wide format
            headers = [
                'timestep', 'experience_level', 'state', 'meta_awareness', 'dominant_thoughtseed'
            ]
            headers.extend([f'ts_{ts}' for ts in learner.thoughtseeds])
            
            if hasattr(learner, 'network_activations_history'):
                headers.extend([f'net_{net}' for net in learner.networks])
                headers.extend(['free_energy', 'prediction_error', 'precision'])
            
            writer.writerow(headers)
            
            # Data rows
            for t in range(len(learner.activations_history)):
                row = [
                    t,
                    learner.experience_level,
                    learner.state_history[t],
                    learner.meta_awareness_history[t],
                    learner.dominant_ts_history[t]
                ]
                
                # Thoughtseed activations
                row.extend(learner.activations_history[t])
                
                # Network data if available
                if hasattr(learner, 'network_activations_history'):
                    row.extend([learner.network_activations_history[t][net] for net in learner.networks])
                    row.extend([
                        learner.free_energy_history[t],
                        learner.prediction_error_history[t],
                        learner.precision_history[t]
                    ])
                
                writer.writerow(row)
        
        return filepath
    
    def _export_summary_data(self, learner: Any, base_name: str) -> str:
        """Export summary statistics for R analysis."""
        filepath = self._generate_filename(base_name, "csv", learner.experience_level)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Headers
            writer.writerow(['category', 'item', 'metric', 'value', 'experience_level'])
            
            # Thoughtseed statistics
            activations_array = np.array(learner.activations_history)
            
            for i, ts in enumerate(learner.thoughtseeds):
                ts_data = activations_array[:, i]
                
                stats = [
                    ('mean', np.mean(ts_data)),
                    ('sd', np.std(ts_data)),
                    ('median', np.median(ts_data)),
                    ('min', np.min(ts_data)),
                    ('max', np.max(ts_data)),
                    ('q25', np.percentile(ts_data, 25)),
                    ('q75', np.percentile(ts_data, 75))
                ]
                
                for metric, value in stats:
                    writer.writerow(['thoughtseed', ts, metric, value, learner.experience_level])
            
            # State statistics
            for state in learner.states:
                frequency = learner.state_history.count(state) / len(learner.state_history)
                writer.writerow(['state', state, 'frequency', frequency, learner.experience_level])
                
                # Average duration in state
                durations = self._calculate_state_durations(learner.state_history, state)
                if durations:
                    writer.writerow(['state', state, 'mean_duration', np.mean(durations), learner.experience_level])
                    writer.writerow(['state', state, 'sd_duration', np.std(durations), learner.experience_level])
            
            # Meta-awareness statistics
            ma_stats = [
                ('mean', np.mean(learner.meta_awareness_history)),
                ('sd', np.std(learner.meta_awareness_history)),
                ('median', np.median(learner.meta_awareness_history)),
                ('min', np.min(learner.meta_awareness_history)),
                ('max', np.max(learner.meta_awareness_history))
            ]
            
            for metric, value in ma_stats:
                writer.writerow(['meta_awareness', 'overall', metric, value, learner.experience_level])
            
            # Network statistics if available
            if hasattr(learner, 'network_activations_history'):
                network_array = np.array([
                    [step[net] for net in learner.networks]
                    for step in learner.network_activations_history
                ])
                
                for i, net in enumerate(learner.networks):
                    net_data = network_array[:, i]
                    
                    net_stats = [
                        ('mean', np.mean(net_data)),
                        ('sd', np.std(net_data)),
                        ('median', np.median(net_data))
                    ]
                    
                    for metric, value in net_stats:
                        writer.writerow(['network', net, metric, value, learner.experience_level])
                
                # Free energy statistics
                fe_stats = [
                    ('mean', np.mean(learner.free_energy_history)),
                    ('sd', np.std(learner.free_energy_history)),
                    ('final', learner.free_energy_history[-1])
                ]
                
                for metric, value in fe_stats:
                    writer.writerow(['free_energy', 'overall', metric, value, learner.experience_level])
        
        return filepath
    
    def _export_network_data(self, learner: Any, base_name: str) -> str:
        """Export network-specific analysis data."""
        filepath = self._generate_filename(base_name, "csv", learner.experience_level)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Network correlation data
            network_array = np.array([
                [step[net] for net in learner.networks]
                for step in learner.network_activations_history
            ])
            
            # Headers for correlation matrix
            writer.writerow(['network1', 'network2', 'correlation', 'experience_level'])
            
            # Correlation pairs
            corr_matrix = np.corrcoef(network_array.T)
            
            for i, net1 in enumerate(learner.networks):
                for j, net2 in enumerate(learner.networks):
                    writer.writerow([net1, net2, corr_matrix[i, j], learner.experience_level])
        
        return filepath
    
    def _generate_r_script(self, learner: Any, base_name: str, 
                          result_files: Dict[str, str]) -> str:
        """Generate R analysis script for the exported data."""
        script_filepath = self._generate_filename(f"{base_name}_analysis", "R", learner.experience_level)
        
        script_content = f"""
# Meditation Simulation Analysis Script
# Auto-generated for {learner.experience_level} data
# Generated: {self.metadata['export_timestamp']}

# Load required packages
if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, ggplot2, corrplot, psych, gridExtra)

# Set working directory (adjust as needed)
# setwd("path/to/your/exports")

# Load data
print("Loading meditation simulation data...")

"""
        
        # Add data loading commands
        for data_type, filepath in result_files.items():
            if data_type.endswith('_data'):
                var_name = data_type.replace('_data', '_df')
                script_content += f'{var_name} <- read.csv("{filepath}")\n'
        
        script_content += f"""

# Display basic information
print(paste("Experience Level:", "{learner.experience_level}"))
print(paste("Total timesteps:", {len(learner.activations_history)}))

# Basic descriptive statistics
if (exists("main_df")) {{
  print("=== Data Overview ===")
  print(str(main_df))
  print(summary(main_df))
}}

# Thoughtseed activation plots
if (exists("main_df") && "{self.long_format}" == "True") {{
  
  # Time series plot of thoughtseed activations
  p1 <- main_df %>%
    filter(variable_type == "thoughtseed") %>%
    ggplot(aes(x = timestep, y = value, color = variable)) +
    geom_line() +
    labs(title = "Thoughtseed Activations Over Time",
         x = "Timestep", y = "Activation Level") +
    theme_minimal() +
    guides(color = guide_legend(title = "Thoughtseed"))
  
  print(p1)
  
  # Meta-awareness over time
  p2 <- main_df %>%
    select(timestep, meta_awareness) %>%
    distinct() %>%
    ggplot(aes(x = timestep, y = meta_awareness)) +
    geom_line(color = "blue", size = 1) +
    labs(title = "Meta-Awareness Over Time",
         x = "Timestep", y = "Meta-Awareness Level") +
    theme_minimal()
  
  print(p2)
  
  # State distribution
  p3 <- main_df %>%
    select(timestep, state) %>%
    distinct() %>%
    ggplot(aes(x = state)) +
    geom_bar(fill = "steelblue", alpha = 0.7) +
    labs(title = "Distribution of Meditation States",
         x = "State", y = "Frequency") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  print(p3)
}}

"""
        
        # Add network analysis if applicable
        if hasattr(learner, 'network_activations_history'):
            script_content += """
# Network analysis
if (exists("main_df") && any(main_df$variable_type == "network")) {
  
  # Network activations over time
  p4 <- main_df %>%
    filter(variable_type == "network") %>%
    ggplot(aes(x = timestep, y = value, color = variable)) +
    geom_line() +
    labs(title = "Network Activations Over Time",
         x = "Timestep", y = "Activation Level") +
    theme_minimal() +
    guides(color = guide_legend(title = "Network"))
  
  print(p4)
  
  # Free energy over time
  p5 <- main_df %>%
    filter(variable == "free_energy") %>%
    ggplot(aes(x = timestep, y = value)) +
    geom_line(color = "red", size = 1) +
    labs(title = "Free Energy Over Time",
         x = "Timestep", y = "Free Energy") +
    theme_minimal()
  
  print(p5)
}

# Network correlation analysis
if (exists("network_df")) {
  
  # Correlation heatmap
  corr_matrix <- network_df %>%
    select(network1, network2, correlation) %>%
    pivot_wider(names_from = network2, values_from = correlation) %>%
    column_to_rownames("network1") %>%
    as.matrix()
  
  corrplot(corr_matrix, method = "color", type = "upper",
           title = "Network Activation Correlations")
}
"""
        
        script_content += """
# Summary statistics
if (exists("summary_df")) {
  
  print("=== Summary Statistics ===")
  
  # Thoughtseed statistics
  ts_stats <- summary_df %>%
    filter(category == "thoughtseed") %>%
    select(item, metric, value) %>%
    pivot_wider(names_from = metric, values_from = value)
  
  print("Thoughtseed Statistics:")
  print(ts_stats)
  
  # State frequencies  
  state_freq <- summary_df %>%
    filter(category == "state", metric == "frequency") %>%
    select(item, value)
  
  print("State Frequencies:")
  print(state_freq)
}

print("Analysis complete!")
print("Generated plots and summaries are displayed above.")
"""
        
        with open(script_filepath, 'w') as f:
            f.write(script_content)
        
        return script_filepath
    
    def _calculate_state_durations(self, state_history: List[str], target_state: str) -> List[int]:
        """Calculate durations for each occurrence of a specific state."""
        durations = []
        current_duration = 0
        
        for state in state_history:
            if state == target_state:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        # Add final duration if simulation ended in target state
        if current_duration > 0:
            durations.append(current_duration)
        
        return durations
