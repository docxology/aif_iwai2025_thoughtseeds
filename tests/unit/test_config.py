"""
Comprehensive unit tests for the configuration system.

This module tests all configuration classes, parameters, and validation
logic to ensure consistency with documented specifications.
"""

import pytest
import numpy as np
from dataclasses import fields

from config import (
    ActiveInferenceConfig, THOUGHTSEEDS, STATES,
    ThoughtseedParams, MetacognitionParams,
    DwellTimeConfig, ActiveInferenceParameters,
    TransitionThresholds, NetworkModulationConfig,
    NonLinearDynamicsConfig, THOUGHTSEED_AGENTS,
    NETWORK_PROFILES, STATE_DWELL_TIMES
)


class TestConstants:
    """Test core constants and definitions."""
    
    def test_thoughtseeds_definition(self):
        """Test THOUGHTSEEDS constant."""
        expected = ['breath_focus', 'pain_discomfort', 'pending_tasks', 
                   'self_reflection', 'equanimity']
        assert THOUGHTSEEDS == expected
        assert len(THOUGHTSEEDS) == 5
        assert len(set(THOUGHTSEEDS)) == 5  # No duplicates
    
    def test_states_definition(self):
        """Test STATES constant."""
        expected = ['breath_control', 'mind_wandering', 'meta_awareness', 'redirect_breath']
        assert STATES == expected
        assert len(STATES) == 4
        assert len(set(STATES)) == 4  # No duplicates


class TestThoughtseedAgents:
    """Test thoughtseed agent configuration."""
    
    def test_agent_completeness(self):
        """Test that all thoughtseeds have agent definitions."""
        assert set(THOUGHTSEED_AGENTS.keys()) == set(THOUGHTSEEDS)
    
    def test_agent_structure(self):
        """Test agent data structure."""
        for ts_name, agent in THOUGHTSEED_AGENTS.items():
            # Check required fields
            assert hasattr(agent, 'id')
            assert hasattr(agent, 'category')
            assert hasattr(agent, 'intentional_weights')
            assert hasattr(agent, 'decay_rate')
            assert hasattr(agent, 'recovery_rate')
            
            # Check types
            assert isinstance(agent.id, int)
            assert isinstance(agent.category, str)
            assert isinstance(agent.intentional_weights, dict)
            assert isinstance(agent.decay_rate, (float, int))
            assert isinstance(agent.recovery_rate, (float, int))
            
            # Check intentional weights
            assert 'novice' in agent.intentional_weights
            assert 'expert' in agent.intentional_weights
            assert 0.0 <= agent.intentional_weights['novice'] <= 1.0
            assert 0.0 <= agent.intentional_weights['expert'] <= 1.0
            
            # Check rates are positive
            assert agent.decay_rate > 0
            assert agent.recovery_rate > 0
    
    def test_agent_categories(self):
        """Test agent category assignments."""
        categories = {agent.category for agent in THOUGHTSEED_AGENTS.values()}
        expected_categories = {'focus', 'distraction', 'metacognition', 'regulation'}
        assert categories == expected_categories
        
        # Check specific categorizations
        assert THOUGHTSEED_AGENTS['breath_focus'].category == 'focus'
        assert THOUGHTSEED_AGENTS['pain_discomfort'].category == 'distraction'
        assert THOUGHTSEED_AGENTS['pending_tasks'].category == 'distraction'
        assert THOUGHTSEED_AGENTS['self_reflection'].category == 'metacognition'
        assert THOUGHTSEED_AGENTS['equanimity'].category == 'regulation'
    
    def test_expert_advantages(self):
        """Test that experts have advantages in key thoughtseeds."""
        # Breath focus should be higher for experts
        breath_agent = THOUGHTSEED_AGENTS['breath_focus']
        assert breath_agent.intentional_weights['expert'] > breath_agent.intentional_weights['novice']
        
        # Equanimity should be much higher for experts
        eq_agent = THOUGHTSEED_AGENTS['equanimity']
        assert eq_agent.intentional_weights['expert'] > eq_agent.intentional_weights['novice']
        
        # Self-reflection should be higher for experts
        sr_agent = THOUGHTSEED_AGENTS['self_reflection']
        assert sr_agent.intentional_weights['expert'] > sr_agent.intentional_weights['novice']


class TestNetworkProfiles:
    """Test network profile configurations."""
    
    def test_network_profile_structure(self):
        """Test network profile data structure."""
        assert 'thoughtseed_contributions' in NETWORK_PROFILES
        assert 'state_expected_profiles' in NETWORK_PROFILES
        
        # Check thoughtseed contributions
        ts_contribs = NETWORK_PROFILES['thoughtseed_contributions']
        assert set(ts_contribs.keys()) == set(THOUGHTSEEDS)
        
        for ts_name, profile in ts_contribs.items():
            assert hasattr(profile, 'DMN')
            assert hasattr(profile, 'VAN')
            assert hasattr(profile, 'DAN')
            assert hasattr(profile, 'FPN')
            
            # Check bounds
            assert 0.0 <= profile.DMN <= 1.0
            assert 0.0 <= profile.VAN <= 1.0
            assert 0.0 <= profile.DAN <= 1.0
            assert 0.0 <= profile.FPN <= 1.0
    
    def test_thoughtseed_network_mappings(self):
        """Test biologically plausible thoughtseed-network mappings."""
        contributions = NETWORK_PROFILES['thoughtseed_contributions']
        
        # Breath focus should strongly activate DAN
        breath_profile = contributions['breath_focus']
        assert breath_profile.DAN >= 0.7
        
        # Pending tasks should strongly activate DMN
        tasks_profile = contributions['pending_tasks']
        assert tasks_profile.DMN >= 0.7
        
        # Pain should strongly activate VAN (salience)
        pain_profile = contributions['pain_discomfort']
        assert pain_profile.VAN >= 0.6
        
        # Self-reflection should activate FPN
        reflection_profile = contributions['self_reflection']
        assert reflection_profile.FPN >= 0.7
        
        # Equanimity should strongly activate FPN
        eq_profile = contributions['equanimity']
        assert eq_profile.FPN >= 0.8
    
    def test_state_expected_profiles(self):
        """Test state-specific network expectations."""
        state_profiles = NETWORK_PROFILES['state_expected_profiles']
        
        assert set(state_profiles.keys()) == set(STATES)
        
        for state in STATES:
            assert 'novice' in state_profiles[state]
            assert 'expert' in state_profiles[state]
            
            novice_profile = state_profiles[state]['novice']
            expert_profile = state_profiles[state]['expert']
            
            # Check all networks present
            for profile in [novice_profile, expert_profile]:
                assert hasattr(profile, 'DMN')
                assert hasattr(profile, 'VAN')
                assert hasattr(profile, 'DAN')
                assert hasattr(profile, 'FPN')
    
    def test_expert_network_differences(self):
        """Test expert vs novice network profile differences."""
        state_profiles = NETWORK_PROFILES['state_expected_profiles']
        
        for state in STATES:
            novice = state_profiles[state]['novice']
            expert = state_profiles[state]['expert']
            
            if state in ['breath_control', 'redirect_breath']:
                # Experts should have higher DAN in focused states
                assert expert.DAN >= novice.DAN
                
                # Experts should have lower DMN in focused states  
                assert expert.DMN <= novice.DMN
                
                # Experts should have higher FPN (cognitive control)
                assert expert.FPN >= novice.FPN


class TestDwellTimeConfig:
    """Test dwell time configuration."""
    
    def test_dwell_time_structure(self):
        """Test dwell time data structure."""
        assert 'novice' in STATE_DWELL_TIMES
        assert 'expert' in STATE_DWELL_TIMES
        
        for experience_level in ['novice', 'expert']:
            dwell_config = STATE_DWELL_TIMES[experience_level]
            
            # Check all states present
            for state in STATES:
                assert state in dwell_config
                min_time, max_time = dwell_config[state]
                
                # Check types and validity
                assert isinstance(min_time, int)
                assert isinstance(max_time, int)
                assert min_time > 0
                assert max_time > min_time
                assert max_time <= 50  # Reasonable upper bound
    
    def test_expert_novice_dwell_differences(self):
        """Test experience-level dwell time differences."""
        novice_dwells = STATE_DWELL_TIMES['novice']
        expert_dwells = STATE_DWELL_TIMES['expert']
        
        # Experts should have longer focused states
        for state in ['breath_control', 'redirect_breath']:
            novice_min, novice_max = novice_dwells[state]
            expert_min, expert_max = expert_dwells[state]
            
            # Expert ranges should generally be higher
            assert expert_min >= novice_min or expert_max > novice_max
        
        # Experts should have shorter wandering states
        novice_wander = novice_dwells['mind_wandering']
        expert_wander = expert_dwells['mind_wandering']
        
        # Expert wandering should be shorter or equal
        assert expert_wander[1] <= novice_wander[1]  # Max should be lower
    
    def test_brief_state_durations(self):
        """Test that brief states have appropriate durations."""
        brief_states = ['meta_awareness', 'redirect_breath']
        
        for experience_level in ['novice', 'expert']:
            for state in brief_states:
                min_time, max_time = STATE_DWELL_TIMES[experience_level][state]
                
                # Brief states should be short
                assert max_time <= 10
                assert min_time >= 1


class TestActiveInferenceConfig:
    """Test Active Inference configuration system."""
    
    def test_config_access(self):
        """Test configuration access methods."""
        novice_params = ActiveInferenceConfig.get_params('novice')
        expert_params = ActiveInferenceConfig.get_params('expert')
        
        # Check return types
        assert isinstance(novice_params, dict)
        assert isinstance(expert_params, dict)
        
        # Check required parameters
        required_params = [
            'precision_weight', 'complexity_penalty', 'learning_rate',
            'noise_level', 'memory_factor', 'fpn_enhancement'
        ]
        
        for param in required_params:
            assert param in novice_params
            assert param in expert_params
    
    def test_expert_parameter_advantages(self):
        """Test that expert parameters reflect documented advantages."""
        novice_params = ActiveInferenceConfig.get_params('novice')
        expert_params = ActiveInferenceConfig.get_params('expert')
        
        # Experts should have higher precision weight
        assert expert_params['precision_weight'] >= novice_params['precision_weight']
        
        # Experts should have lower complexity penalty
        assert expert_params['complexity_penalty'] <= novice_params['complexity_penalty']
        
        # Experts should have higher learning rate
        assert expert_params['learning_rate'] >= novice_params['learning_rate']
        
        # Experts should have lower noise level
        assert expert_params['noise_level'] <= novice_params['noise_level']
        
        # Experts should have higher memory factor
        assert expert_params['memory_factor'] >= novice_params['memory_factor']
        
        # Experts should have FPN enhancement
        assert expert_params['fpn_enhancement'] >= 1.0


class TestThoughtseedParams:
    """Test thoughtseed parameter calculation."""
    
    def test_get_target_activations(self):
        """Test target activation calculation."""
        # Test different combinations
        for state in STATES:
            for experience_level in ['novice', 'expert']:
                activations = ThoughtseedParams.get_target_activations(
                    state, meta_awareness=0.5, experience_level=experience_level)
                
                # Check output format
                assert isinstance(activations, dict)
                assert set(activations.keys()) == set(THOUGHTSEEDS)
                
                # Check bounds
                for ts, value in activations.items():
                    assert isinstance(value, (float, int, np.number))
                    assert 0.0 <= value <= 1.5  # Allow some overflow for modulation
    
    def test_meta_awareness_modulation(self):
        """Test meta-awareness effects on target activations."""
        state = 'breath_control'
        experience_level = 'expert'
        
        # Low meta-awareness
        low_ma_acts = ThoughtseedParams.get_target_activations(
            state, meta_awareness=0.1, experience_level=experience_level)
        
        # High meta-awareness
        high_ma_acts = ThoughtseedParams.get_target_activations(
            state, meta_awareness=0.9, experience_level=experience_level)
        
        # Meta-awareness should enhance breath focus and equanimity in focused states
        assert high_ma_acts['breath_focus'] >= low_ma_acts['breath_focus']
        assert high_ma_acts['equanimity'] >= low_ma_acts['equanimity']
    
    def test_expert_adjustments(self):
        """Test expert-specific adjustments."""
        state = 'mind_wandering'
        meta_awareness = 0.5
        
        novice_acts = ThoughtseedParams.get_target_activations(
            state, meta_awareness, experience_level='novice')
        expert_acts = ThoughtseedParams.get_target_activations(
            state, meta_awareness, experience_level='expert')
        
        # Experts should have less task distraction during mind wandering
        assert expert_acts['pending_tasks'] <= novice_acts['pending_tasks']
        
        # Experts should have less pain distraction
        assert expert_acts['pain_discomfort'] <= novice_acts['pain_discomfort']


class TestMetacognitionParams:
    """Test metacognition parameter system."""
    
    def test_calculate_meta_awareness(self):
        """Test meta-awareness calculation."""
        thoughtseed_activations = {
            'breath_focus': 0.7,
            'pain_discomfort': 0.2,
            'pending_tasks': 0.1,
            'self_reflection': 0.6,
            'equanimity': 0.5
        }
        
        for state in STATES:
            for experience_level in ['novice', 'expert']:
                meta_awareness = MetacognitionParams.calculate_meta_awareness(
                    state, thoughtseed_activations, experience_level)
                
                # Check bounds
                assert isinstance(meta_awareness, (float, np.floating))
                assert 0.2 <= meta_awareness <= 1.0
    
    def test_thoughtseed_influences(self):
        """Test thoughtseed influences on meta-awareness."""
        state = 'breath_control'
        
        # High self-reflection should increase meta-awareness
        high_reflection = {
            'self_reflection': 0.8,
            'equanimity': 0.3,
            'breath_focus': 0.5,
            'pain_discomfort': 0.2,
            'pending_tasks': 0.1
        }
        
        low_reflection = {
            'self_reflection': 0.2,
            'equanimity': 0.3,
            'breath_focus': 0.5,
            'pain_discomfort': 0.2,
            'pending_tasks': 0.1
        }
        
        high_ma = MetacognitionParams.calculate_meta_awareness(
            state, high_reflection, 'novice')
        low_ma = MetacognitionParams.calculate_meta_awareness(
            state, low_reflection, 'novice')
        
        assert high_ma > low_ma
    
    def test_expert_meta_awareness_advantage(self):
        """Test expert advantages in meta-awareness."""
        state = 'meta_awareness'
        thoughtseed_activations = {
            'self_reflection': 0.6,
            'equanimity': 0.4,
            'breath_focus': 0.3,
            'pain_discomfort': 0.2,
            'pending_tasks': 0.2
        }
        
        novice_ma = MetacognitionParams.calculate_meta_awareness(
            state, thoughtseed_activations, 'novice')
        expert_ma = MetacognitionParams.calculate_meta_awareness(
            state, thoughtseed_activations, 'expert')
        
        # Expert should have higher meta-awareness for same inputs
        assert expert_ma >= novice_ma


class TestTransitionThresholds:
    """Test state transition threshold configuration."""
    
    def test_threshold_creation(self):
        """Test threshold object creation."""
        novice_thresholds = TransitionThresholds.novice()
        expert_thresholds = TransitionThresholds.expert()
        
        # Check all required fields
        for thresholds in [novice_thresholds, expert_thresholds]:
            assert hasattr(thresholds, 'mind_wandering')
            assert hasattr(thresholds, 'dmn_dan_ratio')
            assert hasattr(thresholds, 'meta_awareness')
            assert hasattr(thresholds, 'return_focus')
            
            # Check bounds
            assert 0.0 <= thresholds.mind_wandering <= 1.0
            assert 0.0 <= thresholds.dmn_dan_ratio <= 1.0
            assert 0.0 <= thresholds.meta_awareness <= 1.0
            assert 0.0 <= thresholds.return_focus <= 1.0
    
    def test_expert_threshold_advantages(self):
        """Test expert advantages in thresholds."""
        novice = TransitionThresholds.novice()
        expert = TransitionThresholds.expert()
        
        # Experts should be more resistant to distraction
        assert expert.mind_wandering >= novice.mind_wandering
        
        # Experts should be more sensitive to meta-awareness (lower threshold)
        assert expert.meta_awareness <= novice.meta_awareness
        
        # Experts should return to focus more easily (lower threshold)
        assert expert.return_focus <= novice.return_focus


class TestNetworkModulationConfig:
    """Test network modulation configuration."""
    
    def test_config_creation(self):
        """Test network modulation config creation."""
        novice_config = NetworkModulationConfig.novice()
        expert_config = NetworkModulationConfig.expert()
        
        # Check basic structure
        assert isinstance(novice_config, NetworkModulationConfig)
        assert isinstance(expert_config, NetworkModulationConfig)
        
        # Check that configs have different patterns
        # (Implementation details depend on specific configuration)
        assert novice_config != expert_config


class TestNonLinearDynamicsConfig:
    """Test non-linear dynamics configuration."""
    
    def test_config_structure(self):
        """Test non-linear dynamics config structure."""
        config = NonLinearDynamicsConfig()
        
        # Check compression thresholds
        assert hasattr(config, 'high_compression_threshold')
        assert hasattr(config, 'low_compression_threshold')
        assert hasattr(config, 'high_compression_factor')
        assert hasattr(config, 'low_compression_factor')
        
        # Check bounds
        assert 0.0 < config.high_compression_threshold <= 1.0
        assert 0.0 <= config.low_compression_threshold < config.high_compression_threshold
        assert 0.0 < config.high_compression_factor <= 1.0
        assert 0.0 < config.low_compression_factor <= 1.0


class TestConfigurationConsistency:
    """Test consistency across configuration system."""
    
    def test_thoughtseed_consistency(self):
        """Test that thoughtseed names are consistent across all configs."""
        # Check that all configs reference the same thoughtseeds
        assert set(THOUGHTSEED_AGENTS.keys()) == set(THOUGHTSEEDS)
        
        network_profile_ts = set(NETWORK_PROFILES['thoughtseed_contributions'].keys())
        assert network_profile_ts == set(THOUGHTSEEDS)
    
    def test_state_consistency(self):
        """Test that state names are consistent across all configs."""
        # Check network profiles
        state_profiles = set(NETWORK_PROFILES['state_expected_profiles'].keys())
        assert state_profiles == set(STATES)
        
        # Check dwell times
        for experience_level in STATE_DWELL_TIMES:
            dwell_states = set(STATE_DWELL_TIMES[experience_level].keys())
            assert dwell_states == set(STATES)
    
    def test_experience_level_consistency(self):
        """Test that experience levels are consistently defined."""
        experience_levels = {'novice', 'expert'}
        
        # Check thoughtseed agents
        for agent in THOUGHTSEED_AGENTS.values():
            assert set(agent.intentional_weights.keys()) == experience_levels
        
        # Check state profiles
        for state_profiles in NETWORK_PROFILES['state_expected_profiles'].values():
            assert set(state_profiles.keys()) == experience_levels
        
        # Check dwell times
        assert set(STATE_DWELL_TIMES.keys()) == experience_levels
    
    def test_parameter_ranges(self):
        """Test that all parameters are in reasonable ranges."""
        # Test Active Inference parameters
        for experience_level in ['novice', 'expert']:
            params = ActiveInferenceConfig.get_params(experience_level)
            
            # Precision weight should be reasonable
            assert 0.1 <= params['precision_weight'] <= 1.0
            
            # Complexity penalty should be reasonable
            assert 0.0 <= params['complexity_penalty'] <= 1.0
            
            # Learning rate should be small but positive
            assert 0.001 <= params['learning_rate'] <= 0.1
            
            # Noise level should be small but positive
            assert 0.001 <= params['noise_level'] <= 0.2
            
            # Memory factor should be between 0 and 1
            assert 0.1 <= params['memory_factor'] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
