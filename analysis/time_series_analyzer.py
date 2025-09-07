"""
Time series analysis functionality for meditation simulation data.

This module provides specialized time series analysis tools including
trend detection, periodicity analysis, change point detection, and
temporal pattern characterization for meditation data.
"""

import numpy as np
from typing import Dict, List, Any, Tuple

try:
    from scipy import signal
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Simple substitutes for scipy functions
    class SimpleSignal:
        @staticmethod
        def detrend(data):
            # Simple linear detrend
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data, 1)
            trend = coeffs[0] * x + coeffs[1]
            return data - trend
    
    signal = SimpleSignal()
    
    def pearsonr(x, y):
        return np.corrcoef(x, y)[0, 1], 0.05  # correlation, p-value


class TimeSeriesAnalyzer:
    """
    Specialized time series analyzer for meditation simulation data.
    
    Provides analysis tools for temporal patterns including trends,
    periodicities, change points, and regime shifts in meditation
    time series data.
    """
    
    def __init__(self):
        """Initialize time series analyzer."""
        self.has_scipy = SCIPY_AVAILABLE
        self.has_pandas = False  # TimeSeriesAnalyzer doesn't use pandas
    
    def detrend_data(self, data: List[float]) -> List[float]:
        """Remove trend from time series data."""
        if len(data) < 2:
            return data
        
        data_array = np.array(data)
        detrended = signal.detrend(data_array)
        return detrended.tolist()
    
    def detect_periodicities(self, data: List[float]) -> Dict[str, Any]:
        """Detect periodic patterns in time series."""
        if len(data) < 10:
            return {'periods': [], 'dominant_period': None}
        
        # Simple autocorrelation-based periodicity detection
        data_array = np.array(data)
        autocorr = np.correlate(data_array, data_array, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, min(len(autocorr) - 1, len(data) // 2)):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append(i)
        
        return {
            'periods': peaks[:5],  # Top 5 periods
            'dominant_period': peaks[0] if peaks else None,
            'autocorrelation_strength': float(max(autocorr[1:10])) if len(autocorr) > 10 else 0.0
        }
    
    def analyze_transitions(self, learner: Any) -> Dict[str, Any]:
        """Analyze state transition patterns over time."""
        if not hasattr(learner, 'state_history'):
            return {}
        
        transitions = []
        for i in range(len(learner.state_history) - 1):
            if learner.state_history[i] != learner.state_history[i + 1]:
                transitions.append({
                    'timestep': i,
                    'from_state': learner.state_history[i],
                    'to_state': learner.state_history[i + 1]
                })
        
        return {
            'transition_count': len(transitions),
            'transition_rate': len(transitions) / len(learner.state_history),
            'transitions': transitions
        }
    
    def calculate_stability_metrics(self, learner: Any) -> Dict[str, float]:
        """Calculate temporal stability metrics."""
        activations_array = np.array(learner.activations_history)
        
        # Overall activation stability (inverse of variance)
        overall_stability = 1 / (np.mean(np.var(activations_array, axis=0)) + 1e-6)
        
        # Meta-awareness stability
        ma_stability = 1 / (np.var(learner.meta_awareness_history) + 1e-6)
        
        return {
            'overall_activation_stability': float(overall_stability),
            'meta_awareness_stability': float(ma_stability),
            'composite_stability': float((overall_stability + ma_stability) / 2)
        }
    
    def analyze_convergence(self, learner: Any) -> Dict[str, Any]:
        """Analyze convergence patterns in the data."""
        if hasattr(learner, 'free_energy_history'):
            fe_data = np.array(learner.free_energy_history)
            
            # Simple convergence check
            if len(fe_data) > 10:
                final_portion = fe_data[-len(fe_data)//4:]
                stability_threshold = 0.05 * np.std(fe_data)
                converged = bool(np.std(final_portion) < stability_threshold)  # Ensure Python bool
                
                return {
                    'converged': converged,
                    'final_stability': float(np.std(final_portion)),
                    'convergence_rate': float(fe_data[0] - fe_data[-1]) / len(fe_data) if len(fe_data) > 1 else 0
                }
        
        return {'converged': False, 'final_stability': 0, 'convergence_rate': 0}
    
    def analyze_time_series(self, learner: Any) -> Dict[str, Any]:
        """Comprehensive time series analysis."""
        return {
            'trend_analysis': self._analyze_trends(learner),
            'periodicity_analysis': self._analyze_periodicity(learner),
            'change_point_analysis': self._detect_change_points(learner),
            'regime_analysis': self._analyze_regimes(learner),
            'stationarity_analysis': self._test_stationarity(learner),
            'autocorrelation_analysis': self._analyze_autocorrelations(learner)
        }
    
    def _analyze_trends(self, learner: Any) -> Dict[str, Any]:
        """Analyze trends in time series data."""
        trends = {}
        
        # Analyze thoughtseed trends
        activations = np.array(learner.activations_history)
        for i, ts in enumerate(learner.thoughtseeds):
            ts_data = activations[:, i]
            slope, r_squared = self._calculate_trend(ts_data)
            
            trends[f'{ts}_trend'] = {
                'slope': float(slope),
                'r_squared': float(r_squared),
                'direction': self._classify_trend(slope),
                'significance': 'significant' if abs(slope) > 0.001 and r_squared > 0.1 else 'not_significant'
            }
        
        # Meta-awareness trend
        ma_slope, ma_r_squared = self._calculate_trend(learner.meta_awareness_history)
        trends['meta_awareness_trend'] = {
            'slope': float(ma_slope),
            'r_squared': float(ma_r_squared),
            'direction': self._classify_trend(ma_slope)
        }
        
        # Network trends if available
        if hasattr(learner, 'network_activations_history'):
            for net in learner.networks:
                net_data = [step[net] for step in learner.network_activations_history]
                slope, r_squared = self._calculate_trend(net_data)
                
                trends[f'{net}_network_trend'] = {
                    'slope': float(slope),
                    'r_squared': float(r_squared),
                    'direction': self._classify_trend(slope)
                }
        
        return trends
    
    def _analyze_periodicity(self, learner: Any) -> Dict[str, Any]:
        """Analyze periodic patterns in the data."""
        periodicity = {}
        
        # Analyze thoughtseed periodicity
        activations = np.array(learner.activations_history)
        for i, ts in enumerate(learner.thoughtseeds):
            ts_data = activations[:, i]
            periods = self._detect_periods(ts_data)
            
            periodicity[f'{ts}_periodicity'] = {
                'dominant_periods': periods[:3] if periods else [],
                'is_periodic': len(periods) > 0,
                'periodicity_strength': float(self._calculate_periodicity_strength(ts_data))
            }
        
        return periodicity
    
    def _detect_change_points(self, learner: Any) -> Dict[str, Any]:
        """Detect change points in time series."""
        change_points = {}
        
        # Change points in meta-awareness
        ma_changes = self._find_change_points(learner.meta_awareness_history)
        change_points['meta_awareness_changes'] = {
            'change_points': ma_changes,
            'num_changes': len(ma_changes),
            'change_frequency': len(ma_changes) / len(learner.meta_awareness_history)
        }
        
        # Change points in thoughtseeds
        activations = np.array(learner.activations_history)
        for i, ts in enumerate(learner.thoughtseeds):
            ts_changes = self._find_change_points(activations[:, i])
            change_points[f'{ts}_changes'] = {
                'change_points': ts_changes,
                'num_changes': len(ts_changes)
            }
        
        return change_points
    
    def _analyze_regimes(self, learner: Any) -> Dict[str, Any]:
        """Analyze regime shifts and stable periods."""
        regimes = {}
        
        # Identify stable regimes in meta-awareness
        ma_regimes = self._identify_regimes(learner.meta_awareness_history)
        regimes['meta_awareness_regimes'] = {
            'num_regimes': len(ma_regimes),
            'regime_durations': [r['duration'] for r in ma_regimes],
            'regime_stability': self._calculate_regime_stability(ma_regimes)
        }
        
        return regimes
    
    def _test_stationarity(self, learner: Any) -> Dict[str, Any]:
        """Test for stationarity in time series."""
        stationarity = {}
        
        # Test thoughtseed stationarity
        activations = np.array(learner.activations_history)
        for i, ts in enumerate(learner.thoughtseeds):
            ts_data = activations[:, i]
            is_stationary = self._is_stationary(ts_data)
            
            stationarity[f'{ts}_stationarity'] = {
                'is_stationary': is_stationary,
                'mean_stability': float(self._test_mean_stability(ts_data)),
                'variance_stability': float(self._test_variance_stability(ts_data))
            }
        
        return stationarity
    
    def _analyze_autocorrelations(self, learner: Any) -> Dict[str, Any]:
        """Analyze autocorrelation patterns."""
        autocorrelations = {}
        
        # Autocorrelations for thoughtseeds
        activations = np.array(learner.activations_history)
        for i, ts in enumerate(learner.thoughtseeds):
            ts_data = activations[:, i]
            autocorr_values = self._calculate_autocorrelation_function(ts_data, max_lag=20)
            
            autocorrelations[f'{ts}_autocorr'] = {
                'lag_1_autocorr': float(autocorr_values[1]) if len(autocorr_values) > 1 else 0.0,
                'significant_lags': self._find_significant_lags(autocorr_values),
                'decay_rate': float(self._calculate_decay_rate(autocorr_values))
            }
        
        return autocorrelations
    
    # Utility methods
    def _calculate_trend(self, data: List[float]) -> Tuple[float, float]:
        """Calculate trend slope and R-squared."""
        if len(data) < 3:
            return 0.0, 0.0
        
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        slope = coeffs[0]
        
        # Calculate R-squared
        predicted = slope * x + coeffs[1]
        ss_tot = np.sum((data - np.mean(data)) ** 2)
        ss_res = np.sum((data - predicted) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return slope, max(0.0, r_squared)
    
    def _classify_trend(self, slope: float) -> str:
        """Classify trend direction."""
        if slope > 0.001:
            return 'increasing'
        elif slope < -0.001:
            return 'decreasing'
        else:
            return 'stable'
    
    def _detect_periods(self, data: np.ndarray, min_period: int = 5) -> List[int]:
        """Detect periodic patterns using FFT."""
        if len(data) < 2 * min_period:
            return []
        
        # Remove trend
        detrended = signal.detrend(data)
        
        # FFT analysis
        fft = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(len(data))
        
        # Find dominant frequencies
        power = np.abs(fft) ** 2
        # Ignore DC component and high frequencies
        valid_indices = np.arange(1, len(data) // 2)
        
        # Find peaks in power spectrum
        peak_indices = []
        for i in valid_indices:
            if (power[i] > power[i-1] and power[i] > power[i+1] and 
                power[i] > 0.1 * np.max(power[valid_indices])):
                period = int(1 / abs(freqs[i])) if freqs[i] != 0 else 0
                if period >= min_period:
                    peak_indices.append(period)
        
        return sorted(peak_indices, key=lambda p: power[int(len(data) / p)] if p > 0 else 0, reverse=True)
    
    def _calculate_periodicity_strength(self, data: np.ndarray) -> float:
        """Calculate strength of periodic patterns."""
        if len(data) < 10:
            return 0.0
        
        # Autocorrelation-based periodicity measure
        autocorr = self._calculate_autocorrelation_function(data, max_lag=min(50, len(data)//2))
        
        # Find maximum autocorrelation (excluding lag 0)
        if len(autocorr) > 1:
            max_autocorr = np.max(np.abs(autocorr[1:]))
            return max_autocorr
        
        return 0.0
    
    def _find_change_points(self, data: List[float], threshold: float = 0.2) -> List[int]:
        """Find change points in time series."""
        if len(data) < 10:
            return []
        
        change_points = []
        window_size = max(3, len(data) // 20)
        
        for i in range(window_size, len(data) - window_size):
            before = np.mean(data[i-window_size:i])
            after = np.mean(data[i:i+window_size])
            
            if abs(after - before) > threshold:
                change_points.append(i)
        
        # Remove nearby change points
        filtered_changes = []
        for cp in change_points:
            if not filtered_changes or cp - filtered_changes[-1] > window_size:
                filtered_changes.append(cp)
        
        return filtered_changes
    
    def _identify_regimes(self, data: List[float], stability_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Identify stable regimes in time series."""
        regimes = []
        current_regime_start = 0
        window_size = max(5, len(data) // 20)
        
        for i in range(window_size, len(data), window_size):
            window = data[max(0, i-window_size):i]
            
            if np.std(window) < stability_threshold:
                # Stable period
                if not regimes or regimes[-1]['type'] != 'stable':
                    regimes.append({
                        'type': 'stable',
                        'start': current_regime_start,
                        'end': i,
                        'duration': i - current_regime_start,
                        'mean_value': np.mean(window)
                    })
                else:
                    # Extend current stable regime
                    regimes[-1]['end'] = i
                    regimes[-1]['duration'] = i - regimes[-1]['start']
            else:
                # Unstable period
                if not regimes or regimes[-1]['type'] != 'unstable':
                    regimes.append({
                        'type': 'unstable',
                        'start': current_regime_start,
                        'end': i,
                        'duration': i - current_regime_start,
                        'volatility': np.std(window)
                    })
                else:
                    regimes[-1]['end'] = i
                    regimes[-1]['duration'] = i - regimes[-1]['start']
            
            current_regime_start = i
        
        return regimes
    
    def _calculate_regime_stability(self, regimes: List[Dict[str, Any]]) -> float:
        """Calculate overall regime stability."""
        if not regimes:
            return 0.0
        
        stable_duration = sum(r['duration'] for r in regimes if r['type'] == 'stable')
        total_duration = sum(r['duration'] for r in regimes)
        
        return stable_duration / total_duration if total_duration > 0 else 0.0
    
    def _is_stationary(self, data: np.ndarray, p_value_threshold: float = 0.05) -> bool:
        """Simple stationarity test based on trend and variance."""
        # Test for trend
        slope, _ = self._calculate_trend(data)
        has_trend = abs(slope) > 0.001
        
        # Test for changing variance
        first_half = data[:len(data)//2]
        second_half = data[len(data)//2:]
        
        var1, var2 = np.var(first_half), np.var(second_half)
        variance_stable = abs(var1 - var2) / max(var1, var2, 1e-6) < 0.5
        
        return not has_trend and variance_stable
    
    def _test_mean_stability(self, data: np.ndarray) -> float:
        """Test stability of mean over time."""
        if len(data) < 4:
            return 1.0
        
        first_half = data[:len(data)//2]
        second_half = data[len(data)//2:]
        
        mean_diff = abs(np.mean(first_half) - np.mean(second_half))
        overall_std = np.std(data)
        
        # Normalize by overall variability
        stability = max(0.0, 1.0 - (mean_diff / (overall_std + 1e-6)))
        return stability
    
    def _test_variance_stability(self, data: np.ndarray) -> float:
        """Test stability of variance over time."""
        if len(data) < 4:
            return 1.0
        
        first_half = data[:len(data)//2]
        second_half = data[len(data)//2:]
        
        var1, var2 = np.var(first_half), np.var(second_half)
        var_ratio = min(var1, var2) / max(var1, var2, 1e-6)
        
        return var_ratio
    
    def _calculate_autocorrelation_function(self, data: np.ndarray, max_lag: int = 20) -> List[float]:
        """Calculate autocorrelation function."""
        if len(data) <= max_lag:
            return [1.0]  # Only lag 0
        
        autocorr = [1.0]  # Lag 0 is always 1
        
        for lag in range(1, min(max_lag + 1, len(data))):
            if len(data) > lag:
                corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                autocorr.append(corr if not np.isnan(corr) else 0.0)
        
        return autocorr
    
    def _find_significant_lags(self, autocorr: List[float], threshold: float = 0.2) -> List[int]:
        """Find lags with significant autocorrelation."""
        significant = []
        
        for lag, corr in enumerate(autocorr[1:], 1):  # Skip lag 0
            if abs(corr) > threshold:
                significant.append(lag)
        
        return significant
    
    def _calculate_decay_rate(self, autocorr: List[float]) -> float:
        """Calculate autocorrelation decay rate."""
        if len(autocorr) < 3:
            return 0.0
        
        # Fit exponential decay to autocorrelation function
        lags = np.arange(1, len(autocorr))
        values = np.abs(autocorr[1:])  # Use absolute values
        
        if len(values) == 0 or np.max(values) == 0:
            return 0.0
        
        # Simple linear fit to log values (exponential decay)
        try:
            log_values = np.log(np.maximum(values, 1e-6))
            slope, _ = np.polyfit(lags, log_values, 1)
            decay_rate = -slope  # Positive decay rate
            return max(0.0, decay_rate)
        except:
            return 0.0
