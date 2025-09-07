"""
Statistical Dashboard for Meditation Data Analysis.

This module provides comprehensive statistical visualizations including
distributions, correlations, hypothesis testing, and comparative analytics
for meditation simulation data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import os

# Handle optional dependencies
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Handle optional scipy dependency
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Simple substitutes for scipy stats functions
    class SimpleStats:
        @staticmethod
        def ttest_ind(group1, group2):
            """Simple t-test approximation without scipy."""
            # Basic two-sample t-test approximation
            mean1, mean2 = np.mean(group1), np.mean(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            n1, n2 = len(group1), len(group2)
            
            # Pooled standard error
            se = np.sqrt(var1/n1 + var2/n2)
            t_stat = (mean1 - mean2) / (se + 1e-10)
            
            # Very rough p-value approximation
            p_val = max(0.001, min(0.999, 2 * (1 - np.minimum(0.999, np.abs(t_stat) / 3))))
            
            return t_stat, p_val
        
        @staticmethod  
        def mannwhitneyu(group1, group2, alternative='two-sided'):
            """Simple Mann-Whitney U approximation without scipy."""
            # Very basic rank-sum approximation
            combined = np.concatenate([group1, group2])
            ranks = np.argsort(np.argsort(combined)) + 1
            
            r1 = np.sum(ranks[:len(group1)])
            u1 = r1 - len(group1) * (len(group1) + 1) / 2
            
            # Rough p-value approximation
            expected = len(group1) * len(group2) / 2
            p_val = max(0.001, min(0.999, 2 * np.abs(u1 - expected) / (expected + 1)))
            
            return u1, p_val
        
        @staticmethod
        def norm():
            """Simple normal distribution approximation."""
            class NormDist:
                @staticmethod
                def fit(data):
                    return np.mean(data), np.std(data)
                
                @staticmethod
                def pdf(x, loc, scale):
                    return (1/(scale * np.sqrt(2*np.pi))) * np.exp(-0.5*((x - loc)/scale)**2)
            return NormDist()
    
    stats = SimpleStats()

from .plotting import STATE_COLORS, NETWORK_COLORS, THOUGHTSEED_COLORS, set_plot_style


class StatisticalDashboard:
    """Comprehensive statistical dashboard for meditation data analysis."""
    
    def __init__(self):
        """Initialize statistical dashboard."""
        self.plot_dir = "results_act_inf/plots"
        os.makedirs(self.plot_dir, exist_ok=True)
        
    def create_comprehensive_statistical_report(self, novice_data: Dict[str, Any], 
                                               expert_data: Dict[str, Any], 
                                               save_path: Optional[str] = None):
        """Create comprehensive statistical analysis dashboard."""
        set_plot_style()
        
        fig = plt.figure(figsize=(20, 24))  # Large figure for comprehensive dashboard
        gs = fig.add_gridspec(6, 4, hspace=0.3, wspace=0.3)
        fig.suptitle('Comprehensive Statistical Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # Helper function to safely extract data
        def safe_extract(data, key, default=None):
            return data.get(key, default) if data else default
        
        # Extract key metrics
        nov_fe = safe_extract(novice_data, 'free_energy_history', [])
        exp_fe = safe_extract(expert_data, 'free_energy_history', [])
        nov_ma = safe_extract(novice_data, 'meta_awareness_history', [])
        exp_ma = safe_extract(expert_data, 'meta_awareness_history', [])
        
        # 1. Distribution Comparison (Top Row - 4 subplots)
        metrics = [
            ('free_energy_history', 'Free Energy', 'blue', 'red'),
            ('meta_awareness_history', 'Meta-awareness', 'lightblue', 'lightcoral'),
            ('prediction_error_history', 'Prediction Error', 'green', 'orange'),
            ('precision_history', 'Precision', 'purple', 'brown')
        ]
        
        for i, (key, title, nov_color, exp_color) in enumerate(metrics):
            ax = fig.add_subplot(gs[0, i])
            
            nov_data = safe_extract(novice_data, key, [])
            exp_data = safe_extract(expert_data, key, [])
            
            if nov_data:
                ax.hist(nov_data, bins=25, alpha=0.6, label='Novice', color=nov_color, density=True)
            if exp_data:
                ax.hist(exp_data, bins=25, alpha=0.6, label='Expert', color=exp_color, density=True)
            
            ax.set_title(f'{title} Distribution')
            ax.set_xlabel(title)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Statistical Tests Results (Second Row)
        ax_stats = fig.add_subplot(gs[1, :2])
        
        # Perform statistical tests
        test_results = []
        
        if nov_fe and exp_fe:
            # T-test for free energy
            t_stat, t_pval = stats.ttest_ind(nov_fe, exp_fe)
            u_stat, u_pval = stats.mannwhitneyu(nov_fe, exp_fe, alternative='two-sided')
            
            test_results.extend([
                ('Free Energy t-test', t_stat, t_pval, 'significant' if t_pval < 0.05 else 'not significant'),
                ('Free Energy Mann-Whitney', u_stat, u_pval, 'significant' if u_pval < 0.05 else 'not significant')
            ])
        
        if nov_ma and exp_ma:
            # T-test for meta-awareness
            t_stat_ma, t_pval_ma = stats.ttest_ind(nov_ma, exp_ma)
            test_results.append(
                ('Meta-awareness t-test', t_stat_ma, t_pval_ma, 'significant' if t_pval_ma < 0.05 else 'not significant')
            )
        
        # Create test results table
        if test_results:
            table_data = []
            for test_name, statistic, p_value, significance in test_results:
                table_data.append([test_name, f'{statistic:.4f}', f'{p_value:.4f}', significance])
            
            ax_stats.axis('tight')
            ax_stats.axis('off')
            
            table = ax_stats.table(cellText=table_data,
                                  colLabels=['Test', 'Statistic', 'p-value', 'Significance'],
                                  cellLoc='center',
                                  loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Color code significance
            for i, (_, _, p_val, sig) in enumerate(test_results):
                color = 'lightgreen' if sig == 'significant' else 'lightcoral'
                table[(i+1, 3)].set_facecolor(color)
        
        ax_stats.set_title('Statistical Test Results')
        
        # 3. Effect Sizes (Second Row, right side)
        ax_effect = fig.add_subplot(gs[1, 2:])
        
        effect_sizes = {}
        if nov_fe and exp_fe:
            # Cohen's d for free energy
            pooled_std = np.sqrt(((len(nov_fe)-1)*np.var(nov_fe) + (len(exp_fe)-1)*np.var(exp_fe)) / 
                               (len(nov_fe) + len(exp_fe) - 2))
            cohens_d_fe = (np.mean(exp_fe) - np.mean(nov_fe)) / pooled_std
            effect_sizes['Free Energy'] = cohens_d_fe
        
        if nov_ma and exp_ma:
            # Cohen's d for meta-awareness
            pooled_std_ma = np.sqrt(((len(nov_ma)-1)*np.var(nov_ma) + (len(exp_ma)-1)*np.var(exp_ma)) / 
                                  (len(nov_ma) + len(exp_ma) - 2))
            cohens_d_ma = (np.mean(exp_ma) - np.mean(nov_ma)) / pooled_std_ma
            effect_sizes['Meta-awareness'] = cohens_d_ma
        
        if effect_sizes:
            metrics_names = list(effect_sizes.keys())
            effect_values = list(effect_sizes.values())
            
            colors = ['green' if abs(d) >= 0.8 else 'orange' if abs(d) >= 0.5 else 'red' 
                     for d in effect_values]
            
            bars = ax_effect.bar(metrics_names, effect_values, color=colors, alpha=0.7)
            ax_effect.axhline(y=0, color='black', linestyle='-', alpha=0.8)
            ax_effect.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
            ax_effect.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect') 
            ax_effect.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect')
            
            # Add value labels
            for bar, value in zip(bars, effect_values):
                height = bar.get_height()
                ax_effect.text(bar.get_x() + bar.get_width()/2, height + 0.02 * max(effect_values),
                             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax_effect.set_title('Effect Sizes (Cohen\'s d)')
        ax_effect.set_ylabel('Effect Size')
        ax_effect.legend()
        ax_effect.grid(True, alpha=0.3)
        
        # 4. Correlation Analysis (Third Row)
        correlation_data = {}
        
        for data, label in [(novice_data, 'Novice'), (expert_data, 'Expert')]:
            if not data:
                continue
                
            # Extract multiple metrics for correlation
            metrics_dict = {}
            for key in ['free_energy_history', 'meta_awareness_history', 'prediction_error_history', 'precision_history']:
                values = safe_extract(data, key, [])
                if values:
                    metrics_dict[key.replace('_history', '').replace('_', ' ').title()] = values
            
            if len(metrics_dict) >= 2:
                # Create correlation matrix
                df_data = {}
                min_length = min(len(values) for values in metrics_dict.values())
                for key, values in metrics_dict.items():
                    df_data[key] = values[:min_length]
                
                if min_length > 1:
                    corr_matrix = np.corrcoef([df_data[key] for key in df_data.keys()])
                    correlation_data[label] = (list(df_data.keys()), corr_matrix)
        
        # Plot correlation matrices
        for i, (label, (keys, corr_matrix)) in enumerate(correlation_data.items()):
            ax = fig.add_subplot(gs[2, i*2:(i+1)*2])
            
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(len(keys)))
            ax.set_yticks(range(len(keys)))
            ax.set_xticklabels(keys, rotation=45)
            ax.set_yticklabels(keys)
            
            # Add correlation values
            for ii in range(len(keys)):
                for jj in range(len(keys)):
                    ax.text(jj, ii, f'{corr_matrix[ii, jj]:.2f}',
                           ha='center', va='center', fontweight='bold',
                           color='white' if abs(corr_matrix[ii, jj]) > 0.5 else 'black')
            
            ax.set_title(f'{label} Correlation Matrix')
            plt.colorbar(im, ax=ax)
        
        # 5. Time Series Analysis (Fourth Row)
        # Stationarity and trend analysis
        ax_trend = fig.add_subplot(gs[3, :2])
        
        for data, label, color in [(novice_data, 'Novice', 'blue'), (expert_data, 'Expert', 'red')]:
            fe_data = safe_extract(data, 'free_energy_history', [])
            if fe_data and len(fe_data) > 10:
                # Detrend and show original vs detrended
                time_steps = np.arange(len(fe_data))
                
                # Linear detrend
                z = np.polyfit(time_steps, fe_data, 1)
                trend = np.poly1d(z)
                detrended = fe_data - trend(time_steps)
                
                ax_trend.plot(time_steps, fe_data, color=color, linewidth=2, 
                             label=f'{label} Original', alpha=0.8)
                ax_trend.plot(time_steps, trend(time_steps), color=color, linewidth=2,
                             linestyle='--', label=f'{label} Trend')
        
        ax_trend.set_title('Trend Analysis')
        ax_trend.set_xlabel('Time Step')
        ax_trend.set_ylabel('Free Energy')
        ax_trend.legend()
        ax_trend.grid(True, alpha=0.3)
        
        # 6. Spectral Analysis (Fourth Row, right)
        ax_spectral = fig.add_subplot(gs[3, 2:])
        
        for data, label, color in [(novice_data, 'Novice', 'blue'), (expert_data, 'Expert', 'red')]:
            fe_data = safe_extract(data, 'free_energy_history', [])
            if fe_data and len(fe_data) > 10:
                # Simple FFT analysis
                fft = np.fft.fft(fe_data)
                freqs = np.fft.fftfreq(len(fe_data))
                
                # Plot power spectrum (positive frequencies only)
                pos_freqs = freqs[:len(freqs)//2]
                power = np.abs(fft[:len(fft)//2])**2
                
                ax_spectral.plot(pos_freqs[1:], power[1:], color=color, linewidth=2, 
                               label=f'{label}', alpha=0.8)
        
        ax_spectral.set_title('Power Spectral Density')
        ax_spectral.set_xlabel('Frequency')
        ax_spectral.set_ylabel('Power')
        ax_spectral.set_yscale('log')
        ax_spectral.legend()
        ax_spectral.grid(True, alpha=0.3)
        
        # 7. Network Analysis (Fifth Row)
        # Network correlation and connectivity analysis
        for i, (data, title) in enumerate([(novice_data, 'Novice Network Analysis'), 
                                          (expert_data, 'Expert Network Analysis')]):
            ax = fig.add_subplot(gs[4, i*2:(i+1)*2])
            
            net_history = safe_extract(data, 'network_activations_history', [])
            if net_history:
                networks = ['DMN', 'VAN', 'DAN', 'FPN']
                
                # Calculate network metrics
                net_means = {}
                net_stds = {}
                
                for net in networks:
                    values = [step.get(net, 0) for step in net_history]
                    net_means[net] = np.mean(values)
                    net_stds[net] = np.std(values)
                
                # Create network statistics plot
                x_pos = np.arange(len(networks))
                means = [net_means[net] for net in networks]
                stds = [net_stds[net] for net in networks]
                colors = [NETWORK_COLORS[net] for net in networks]
                
                bars = ax.bar(x_pos, means, yerr=stds, color=colors, alpha=0.7, 
                             capsize=5, error_kw={'linewidth': 2})
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(networks)
                ax.set_title(title)
                ax.set_ylabel('Activation Level')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, mean, std in zip(bars, means, stds):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                           f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 8. Thoughtseed Competition Analysis (Sixth Row)
        ax_ts_comp = fig.add_subplot(gs[5, :2])
        
        thoughtseeds = ['breath_focus', 'equanimity', 'self_reflection', 'pain_discomfort', 'pending_tasks']
        
        # Calculate thoughtseed statistics
        ts_stats = {}
        for data, label in [(novice_data, 'Novice'), (expert_data, 'Expert')]:
            activations = safe_extract(data, 'activations_history', [])
            if activations:
                ts_means = []
                for i, ts in enumerate(thoughtseeds):
                    if i < len(activations[0]):
                        values = [step[i] if i < len(step) else 0 for step in activations]
                        ts_means.append(np.mean(values))
                    else:
                        ts_means.append(0)
                ts_stats[label] = ts_means
        
        if ts_stats:
            x_pos = np.arange(len(thoughtseeds))
            width = 0.35
            
            if 'Novice' in ts_stats:
                bars1 = ax_ts_comp.bar(x_pos - width/2, ts_stats['Novice'], width,
                                      label='Novice', alpha=0.7, 
                                      color=[THOUGHTSEED_COLORS.get(ts, '#888888') for ts in thoughtseeds])
            
            if 'Expert' in ts_stats:
                bars2 = ax_ts_comp.bar(x_pos + width/2, ts_stats['Expert'], width,
                                      label='Expert', alpha=0.7,
                                      color=[THOUGHTSEED_COLORS.get(ts, '#888888') for ts in thoughtseeds],
                                      hatch='//')
            
            ax_ts_comp.set_xticks(x_pos)
            ax_ts_comp.set_xticklabels([ts.replace('_', ' ').title() for ts in thoughtseeds], rotation=45)
            ax_ts_comp.set_title('Average Thoughtseed Activation')
            ax_ts_comp.set_ylabel('Activation Level')
            ax_ts_comp.legend()
            ax_ts_comp.grid(True, alpha=0.3)
        
        # 9. Performance Summary (Sixth Row, right)
        ax_summary = fig.add_subplot(gs[5, 2:])
        
        # Calculate key performance indicators
        performance_metrics = {}
        
        for data, label in [(novice_data, 'Novice'), (expert_data, 'Expert')]:
            metrics = {}
            
            # Free energy efficiency (lower is better)
            fe_data = safe_extract(data, 'free_energy_history', [])
            if fe_data:
                metrics['Energy Efficiency'] = 1 / (np.mean(fe_data) + 1e-6)
            
            # Meta-awareness level (higher is better)
            ma_data = safe_extract(data, 'meta_awareness_history', [])
            if ma_data:
                metrics['Meta-awareness'] = np.mean(ma_data)
            
            # Attention stability (lower variance is better)
            if fe_data:
                metrics['Stability'] = 1 / (np.std(fe_data) + 1e-6)
            
            performance_metrics[label] = metrics
        
        if performance_metrics and len(performance_metrics) > 1:
            # Normalize metrics for radar chart
            all_metrics = set()
            for metrics in performance_metrics.values():
                all_metrics.update(metrics.keys())
            all_metrics = list(all_metrics)
            
            if all_metrics:
                # Create radar chart
                angles = np.linspace(0, 2*np.pi, len(all_metrics), endpoint=False).tolist()
                angles += [angles[0]]  # Close the polygon
                
                ax_summary = plt.subplot(gs[5, 2:], projection='polar')
                
                for label, color in [('Novice', 'blue'), ('Expert', 'red')]:
                    if label in performance_metrics:
                        values = []
                        for metric in all_metrics:
                            if metric in performance_metrics[label]:
                                values.append(performance_metrics[label][metric])
                            else:
                                values.append(0)
                        
                        # Normalize to 0-1 scale
                        max_val = max(max(performance_metrics[l].get(m, 0) for l in performance_metrics.keys()) 
                                    for m in all_metrics)
                        norm_values = [v / (max_val + 1e-6) for v in values]
                        norm_values += [norm_values[0]]  # Close polygon
                        
                        ax_summary.plot(angles, norm_values, color=color, linewidth=2, 
                                       label=label, alpha=0.8)
                        ax_summary.fill(angles, norm_values, color=color, alpha=0.2)
                
                ax_summary.set_xticks(angles[:-1])
                ax_summary.set_xticklabels(all_metrics)
                ax_summary.set_ylim(0, 1)
                ax_summary.set_title('Performance Comparison', pad=20)
                ax_summary.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.plot_dir}/statistical_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_distribution_analysis(self, learner_data: Dict[str, Any], save_path: Optional[str] = None):
        """Create detailed distribution analysis for a single learner."""
        set_plot_style()
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Distribution Analysis - {learner_data.get("experience_level", "").title()}',
                     fontsize=16, fontweight='bold')
        
        # Data keys to analyze
        data_keys = [
            ('free_energy_history', 'Free Energy'),
            ('meta_awareness_history', 'Meta-awareness'),
            ('prediction_error_history', 'Prediction Error'),
            ('precision_history', 'Precision'),
        ]
        
        # Network data
        network_data = learner_data.get('network_activations_history', [])
        if network_data:
            networks = ['DMN', 'VAN', 'DAN', 'FPN']
            for net in networks:
                net_values = [step.get(net, 0) for step in network_data]
                data_keys.append((net_values, f'{net} Network'))
        
        # Limit to 9 subplots
        data_keys = data_keys[:9]
        
        for i, (key, title) in enumerate(data_keys):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            if isinstance(key, str):
                data = learner_data.get(key, [])
            else:
                data = key  # Already extracted network data
            
            if not data:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(title)
                continue
            
            # Histogram with fitted distributions
            ax.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Fit normal distribution
            if SCIPY_AVAILABLE:
                mu, sigma = stats.norm.fit(data)
                x_range = np.linspace(min(data), max(data), 100)
                ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'r-', 
                       linewidth=2, label=f'Normal (μ={mu:.3f}, σ={sigma:.3f})')
            else:
                # Simple normal approximation without scipy
                mu, sigma = np.mean(data), np.std(data)
                x_range = np.linspace(min(data), max(data), 100)
                y_range = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_range - mu)/sigma)**2)
                ax.plot(x_range, y_range, 'r-', 
                       linewidth=2, label=f'Normal (μ={mu:.3f}, σ={sigma:.3f})')
            
            # Add statistical annotations
            ax.axvline(np.mean(data), color='red', linestyle='--', alpha=0.8, label='Mean')
            ax.axvline(np.median(data), color='green', linestyle='--', alpha=0.8, label='Median')
            
            # Calculate and display key statistics
            if SCIPY_AVAILABLE:
                skewness = stats.skew(data)
                kurtosis = stats.kurtosis(data)
            else:
                # Simple skewness and kurtosis approximations
                mean_val = np.mean(data)
                std_val = np.std(data)
                skewness = np.mean(((data - mean_val) / std_val) ** 3) if std_val > 0 else 0
                kurtosis = np.mean(((data - mean_val) / std_val) ** 4) - 3 if std_val > 0 else 0
            
            ax.text(0.02, 0.98, f'Skew: {skewness:.3f}\nKurt: {kurtosis:.3f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(title)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(data_keys), 9):
            row, col = i // 3, i % 3
            axes[row, col].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.plot_dir}/distribution_analysis_{learner_data.get('experience_level', 'unknown')}.png",
                       dpi=300, bbox_inches='tight')
        plt.close()
