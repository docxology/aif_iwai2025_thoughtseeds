"""
Comprehensive unit tests for core learner classes.

This module tests all core functionality in ActInfLearner and RuleBasedLearner,
validating the implementation against the documented specifications.
"""

import pytest
import numpy as np
import os
from unittest.mock import Mock, patch, MagicMock

from core import ActInfLearner, RuleBasedLearner
from config import THOUGHTSEEDS, STATES


class TestRuleBasedLearner:
    """Test the RuleBasedLearner foundation class."""
    
    def test_initialization_novice(self):
        """Test novice learner initialization."""
        learner = RuleBasedLearner(experience_level='novice', timesteps_per_cycle=100)
        
        assert learner.experience_level == 'novice'
        assert learner.timesteps == 100
        assert learner.thoughtseeds == THOUGHTSEEDS
        assert learner.states == STATES
        assert learner.num_thoughtseeds == len(THOUGHTSEEDS)
        assert learner.noise_level > 0  # Should have noise
        
        # Check history initialization
        assert learner.activations_history == []
        assert learner.state_history == []
        assert learner.meta_awareness_history == []
        assert learner.natural_transition_count == 0
        assert learner.forced_transition_count == 0
    
    def test_initialization_expert(self):
        """Test expert learner initialization."""
        learner = RuleBasedLearner(experience_level='expert', timesteps_per_cycle=200)
        
        assert learner.experience_level == 'expert'
        assert learner.timesteps == 200
        # Expert should have lower noise than novice
        novice = RuleBasedLearner(experience_level='novice')
        assert learner.noise_level < novice.noise_level
    
    def test_get_target_activations(self):
        """Test target activation generation."""
        learner = RuleBasedLearner(experience_level='novice')
        
        # Test different states
        for state in STATES:
            activations = learner.get_target_activations(state, meta_awareness=0.5)
            
            # Check output format
            assert isinstance(activations, np.ndarray)
            assert len(activations) == len(THOUGHTSEEDS)
            
            # Check bounds
            assert np.all(activations >= 0.05)
            assert np.all(activations <= 1.0)
            
            # Check state-specific patterns
            if state == 'breath_control':
                breath_idx = THOUGHTSEEDS.index('breath_focus')
                assert activations[breath_idx] > 0.5  # Should be high
                
            elif state == 'mind_wandering':
                task_idx = THOUGHTSEEDS.index('pending_tasks')
                pain_idx = THOUGHTSEEDS.index('pain_discomfort')
                assert activations[task_idx] > 0.4  # Should be high
                assert activations[pain_idx] > 0.4  # Should be high
    
    def test_get_target_activations_meta_awareness_influence(self):
        """Test meta-awareness influence on target activations."""
        learner = RuleBasedLearner(experience_level='expert')
        
        state = 'breath_control'
        
        # Low meta-awareness
        low_ma = learner.get_target_activations(state, meta_awareness=0.1)
        
        # High meta-awareness
        high_ma = learner.get_target_activations(state, meta_awareness=0.8)
        
        # Meta-awareness should generally enhance positive thoughtseeds
        breath_idx = THOUGHTSEEDS.index('breath_focus')
        eq_idx = THOUGHTSEEDS.index('equanimity')
        
        # These should be enhanced by meta-awareness
        assert high_ma[breath_idx] >= low_ma[breath_idx]
        assert high_ma[eq_idx] >= low_ma[eq_idx]
    
    def test_expert_vs_novice_target_activations(self):
        """Test experience-level differences in target activations."""
        state = 'breath_control'
        meta_awareness = 0.5
        
        novice = RuleBasedLearner(experience_level='novice')
        expert = RuleBasedLearner(experience_level='expert')
        
        novice_acts = novice.get_target_activations(state, meta_awareness)
        expert_acts = expert.get_target_activations(state, meta_awareness)
        
        # Experts should have advantages
        breath_idx = THOUGHTSEEDS.index('breath_focus')
        eq_idx = THOUGHTSEEDS.index('equanimity')
        
        # Experts should have higher focus and equanimity
        assert expert_acts[breath_idx] >= novice_acts[breath_idx]
        assert expert_acts[eq_idx] >= novice_acts[eq_idx]
    
    def test_get_dwell_time(self):
        """Test dwell time generation."""
        learner = RuleBasedLearner(experience_level='novice')
        
        for state in STATES:
            dwell_time = learner.get_dwell_time(state)
            
            # Should return positive integer
            assert isinstance(dwell_time, (int, np.integer))
            assert dwell_time >= 1
            
            # Should be within reasonable bounds
            assert dwell_time <= 50  # Reasonable upper limit
            
            # Brief states should have shorter dwell times
            if state in ['meta_awareness', 'redirect_breath']:
                assert dwell_time <= 10  # Brief states
    
    def test_expert_vs_novice_dwell_times(self):
        """Test experience differences in dwell times."""
        novice = RuleBasedLearner(experience_level='novice')
        expert = RuleBasedLearner(experience_level='expert')
        
        # Sample multiple times to get average patterns
        novice_breath = [novice.get_dwell_time('breath_control') for _ in range(20)]
        expert_breath = [expert.get_dwell_time('breath_control') for _ in range(20)]
        
        novice_wander = [novice.get_dwell_time('mind_wandering') for _ in range(20)]
        expert_wander = [expert.get_dwell_time('mind_wandering') for _ in range(20)]
        
        # Experts should have longer focused states on average
        assert np.mean(expert_breath) >= np.mean(novice_breath)
        
        # Experts should have shorter wandering states on average
        assert np.mean(expert_wander) <= np.mean(novice_wander)
    
    def test_get_meta_awareness(self):
        """Test meta-awareness calculation."""
        learner = RuleBasedLearner(experience_level='novice')
        
        # Create test activations
        activations = np.array([0.7, 0.2, 0.1, 0.6, 0.8])  # High self-reflection and equanimity
        
        for state in STATES:
            meta_awareness = learner.get_meta_awareness(state, activations)
            
            # Check output bounds
            assert isinstance(meta_awareness, (float, np.floating))
            assert 0.2 <= meta_awareness <= 0.9
    
    def test_expert_vs_novice_meta_awareness(self):
        """Test experience differences in meta-awareness."""
        activations = np.array([0.5, 0.3, 0.2, 0.7, 0.6])  # Moderate self-reflection
        state = 'breath_control'
        
        novice = RuleBasedLearner(experience_level='novice')
        expert = RuleBasedLearner(experience_level='expert')
        
        novice_ma = novice.get_meta_awareness(state, activations)
        expert_ma = expert.get_meta_awareness(state, activations)
        
        # Experts should have higher meta-awareness
        assert expert_ma >= novice_ma


class TestActInfLearner:
    """Test the ActInfLearner Active Inference implementation."""
    
    def test_initialization(self):
        """Test ActInfLearner initialization."""
        learner = ActInfLearner(experience_level='expert', timesteps_per_cycle=150)
        
        # Check inheritance
        assert isinstance(learner, RuleBasedLearner)
        
        # Check Active Inference specific attributes
        assert hasattr(learner, 'networks')
        assert learner.networks == ['DMN', 'VAN', 'DAN', 'FPN']
        
        assert hasattr(learner, 'precision_weight')
        assert hasattr(learner, 'complexity_penalty')
        assert hasattr(learner, 'learning_rate')
        assert hasattr(learner, 'noise_level')
        assert hasattr(learner, 'memory_factor')
        
        # Check history tracking
        assert hasattr(learner, 'network_activations_history')
        assert hasattr(learner, 'free_energy_history')
        assert hasattr(learner, 'prediction_error_history')
        assert hasattr(learner, 'precision_history')
        
        # Check learned profiles
        assert hasattr(learner, 'learned_network_profiles')
        assert 'thoughtseed_contributions' in learner.learned_network_profiles
        assert 'state_network_expectations' in learner.learned_network_profiles
    
    def test_experience_level_parameters(self):
        """Test experience-level specific parameters."""
        novice = ActInfLearner(experience_level='novice')
        expert = ActInfLearner(experience_level='expert')
        
        # Experts should have higher precision weight
        assert expert.precision_weight >= novice.precision_weight
        
        # Experts should have lower complexity penalty
        assert expert.complexity_penalty <= novice.complexity_penalty
        
        # Experts should have higher learning rate
        assert expert.learning_rate >= novice.learning_rate
        
        # Experts should have lower noise level
        assert expert.noise_level <= novice.noise_level
        
        # Experts should have higher memory factor
        assert expert.memory_factor >= novice.memory_factor
    
    def test_compute_network_activations(self):
        """Test network activation computation."""
        learner = ActInfLearner(experience_level='novice')
        
        # Create test inputs
        thoughtseed_activations = np.array([0.7, 0.2, 0.1, 0.5, 0.6])
        current_state = 'breath_control'
        meta_awareness = 0.5
        
        network_acts = learner.compute_network_activations(
            thoughtseed_activations, current_state, meta_awareness)
        
        # Check output format
        assert isinstance(network_acts, dict)
        assert set(network_acts.keys()) == set(learner.networks)
        
        # Check bounds
        for net, value in network_acts.items():
            assert isinstance(value, (float, np.floating))
            assert 0.05 <= value <= 1.0
        
        # Check state-specific patterns for breath control
        # DAN should be relatively high during focused attention
        assert network_acts['DAN'] > 0.3
        
        # DMN should be suppressed during focused attention
        assert network_acts['DMN'] < 0.7
    
    def test_network_activations_state_specificity(self):
        """Test state-specific network activation patterns."""
        learner = ActInfLearner(experience_level='expert')
        thoughtseed_activations = np.array([0.5, 0.3, 0.2, 0.4, 0.6])
        meta_awareness = 0.6
        
        # Test different states
        breath_nets = learner.compute_network_activations(
            thoughtseed_activations, 'breath_control', meta_awareness)
        
        wander_nets = learner.compute_network_activations(
            thoughtseed_activations, 'mind_wandering', meta_awareness)
        
        meta_nets = learner.compute_network_activations(
            thoughtseed_activations, 'meta_awareness', meta_awareness)
        
        # DMN should be highest during mind wandering
        assert wander_nets['DMN'] >= breath_nets['DMN']
        assert wander_nets['DMN'] >= meta_nets['DMN']
        
        # DAN should be highest during breath control
        assert breath_nets['DAN'] >= wander_nets['DAN']
        
        # VAN should be highest during meta-awareness
        assert meta_nets['VAN'] >= breath_nets['VAN']
        assert meta_nets['VAN'] >= wander_nets['VAN']
    
    def test_calculate_free_energy(self):
        """Test free energy calculation."""
        learner = ActInfLearner(experience_level='novice')
        
        # Initialize learned profiles with some data
        for state in STATES:
            learner.learned_network_profiles["state_network_expectations"][state] = {
                net: 0.5 for net in learner.networks
            }
        
        network_acts = {'DMN': 0.4, 'VAN': 0.6, 'DAN': 0.7, 'FPN': 0.5}
        current_state = 'breath_control'
        meta_awareness = 0.5
        
        free_energy, prediction_errors, total_pe = learner.calculate_free_energy(
            network_acts, current_state, meta_awareness)
        
        # Check outputs
        assert isinstance(free_energy, (float, np.floating))
        assert isinstance(prediction_errors, dict)
        assert isinstance(total_pe, (float, np.floating))
        
        # Free energy should be positive
        assert free_energy > 0
        
        # Prediction errors should exist for all networks
        assert set(prediction_errors.keys()) == set(learner.networks)
        
        # All prediction errors should be non-negative
        for pe in prediction_errors.values():
            assert pe >= 0
    
    def test_free_energy_components(self):
        """Test free energy component contributions."""
        learner = ActInfLearner(experience_level='expert')
        
        # Initialize learned profiles
        for state in STATES:
            learner.learned_network_profiles["state_network_expectations"][state] = {
                net: 0.5 for net in learner.networks
            }
        
        # Test with perfect prediction (zero error)
        network_acts = {'DMN': 0.5, 'VAN': 0.5, 'DAN': 0.5, 'FPN': 0.5}
        fe_perfect, _, _ = learner.calculate_free_energy(
            network_acts, 'breath_control', 0.5)
        
        # Test with large prediction error
        network_acts_error = {'DMN': 0.1, 'VAN': 0.9, 'DAN': 0.1, 'FPN': 0.9}
        fe_error, _, _ = learner.calculate_free_energy(
            network_acts_error, 'breath_control', 0.5)
        
        # Free energy should be higher with larger prediction errors
        assert fe_error > fe_perfect
    
    def test_update_network_profiles(self):
        """Test network profile learning."""
        learner = ActInfLearner(experience_level='expert')
        
        # Initialize some history for precision calculation
        learner.network_activations_history = [{'DMN': 0.5, 'VAN': 0.5, 'DAN': 0.5, 'FPN': 0.5}] * 5
        
        thoughtseed_activations = np.array([0.8, 0.2, 0.1, 0.4, 0.6])  # High breath focus
        network_activations = {'DMN': 0.3, 'VAN': 0.4, 'DAN': 0.8, 'FPN': 0.6}
        current_state = 'breath_control'
        prediction_errors = {'DMN': 0.1, 'VAN': 0.1, 'DAN': 0.1, 'FPN': 0.1}
        
        # Store initial profile
        initial_profile = learner.learned_network_profiles["thoughtseed_contributions"]["breath_focus"]["DAN"]
        
        # Update profiles
        learner.update_network_profiles(
            thoughtseed_activations, network_activations, current_state, prediction_errors)
        
        # Profile should have changed (learning occurred)
        updated_profile = learner.learned_network_profiles["thoughtseed_contributions"]["breath_focus"]["DAN"]
        assert updated_profile != initial_profile
        
        # Profile should remain within bounds
        assert 0.1 <= updated_profile <= 0.9
    
    def test_network_modulated_activations(self):
        """Test network-based thoughtseed modulation."""
        learner = ActInfLearner(experience_level='expert')
        
        initial_activations = np.array([0.5, 0.3, 0.2, 0.4, 0.5])
        network_acts = {'DMN': 0.8, 'VAN': 0.4, 'DAN': 0.3, 'FPN': 0.6}  # High DMN
        current_state = 'mind_wandering'
        
        modulated_acts = learner.network_modulated_activations(
            initial_activations, network_acts, current_state)
        
        # Should return numpy array of same length
        assert isinstance(modulated_acts, np.ndarray)
        assert len(modulated_acts) == len(initial_activations)
        
        # All values should remain in bounds
        assert np.all(modulated_acts >= 0.05)
        assert np.all(modulated_acts <= 1.0)
        
        # High DMN should enhance pending tasks
        task_idx = THOUGHTSEEDS.index('pending_tasks')
        assert modulated_acts[task_idx] >= initial_activations[task_idx]
    
    @pytest.mark.slow
    def test_train_method(self, temp_dir):
        """Test the full training method."""
        # Change to temp directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            learner = ActInfLearner(experience_level='novice', timesteps_per_cycle=30)
            
            # Train the learner
            learner.train()
            
            # Check that histories were populated
            assert len(learner.activations_history) == learner.timesteps
            assert len(learner.network_activations_history) == learner.timesteps
            assert len(learner.state_history) == learner.timesteps
            assert len(learner.free_energy_history) == learner.timesteps
            
            # Check that some transitions occurred
            assert len(set(learner.state_history)) > 1
            
            # Check that natural transitions occurred
            assert learner.natural_transition_count > 0
            
        finally:
            os.chdir(original_dir)
    
    def test_dmn_dan_anticorrelation(self):
        """Test DMN-DAN anticorrelation mechanism."""
        learner = ActInfLearner(experience_level='expert')
        
        # Create activations with high DAN
        thoughtseed_activations = np.array([0.8, 0.1, 0.1, 0.3, 0.5])  # High breath focus
        
        network_acts = learner.compute_network_activations(
            thoughtseed_activations, 'breath_control', 0.6)
        
        # In experts during breath control, DMN should be suppressed when DAN is high
        if network_acts['DAN'] > 0.6:
            assert network_acts['DMN'] < 0.5  # Should be anticorrelated
    
    def test_expert_dmn_suppression(self):
        """Test enhanced DMN suppression in experts."""
        novice = ActInfLearner(experience_level='novice')
        expert = ActInfLearner(experience_level='expert')
        
        # Same inputs
        thoughtseed_activations = np.array([0.7, 0.2, 0.1, 0.4, 0.6])
        current_state = 'breath_control'
        meta_awareness = 0.7
        
        novice_nets = novice.compute_network_activations(
            thoughtseed_activations, current_state, meta_awareness)
        
        expert_nets = expert.compute_network_activations(
            thoughtseed_activations, current_state, meta_awareness)
        
        # Expert should have more DMN suppression during focused states
        assert expert_nets['DMN'] <= novice_nets['DMN']
        
        # Expert should have higher DAN during focused states
        assert expert_nets['DAN'] >= novice_nets['DAN']


class TestLearnerIntegration:
    """Test integration between RuleBasedLearner and ActInfLearner."""
    
    def test_inheritance_compatibility(self):
        """Test that ActInfLearner properly inherits from RuleBasedLearner."""
        learner = ActInfLearner(experience_level='expert')
        
        # Should have all RuleBasedLearner methods
        assert hasattr(learner, 'get_target_activations')
        assert hasattr(learner, 'get_dwell_time')
        assert hasattr(learner, 'get_meta_awareness')
        
        # Should be able to call parent methods
        activations = learner.get_target_activations('breath_control', 0.5)
        assert isinstance(activations, np.ndarray)
        
        dwell_time = learner.get_dwell_time('mind_wandering')
        assert isinstance(dwell_time, (int, np.integer))
        
        meta_awareness = learner.get_meta_awareness('meta_awareness', activations)
        assert isinstance(meta_awareness, (float, np.floating))
    
    def test_method_override_compatibility(self):
        """Test that overridden methods maintain compatibility."""
        base_learner = RuleBasedLearner(experience_level='novice')
        act_inf_learner = ActInfLearner(experience_level='novice')
        
        # Same inputs should produce similar base behavior
        state = 'breath_control'
        meta_awareness = 0.5
        
        base_activations = base_learner.get_target_activations(state, meta_awareness)
        ai_activations = act_inf_learner.get_target_activations(state, meta_awareness)
        
        # Should have same format and bounds
        assert base_activations.shape == ai_activations.shape
        assert np.all(base_activations >= 0.05) and np.all(ai_activations >= 0.05)
        assert np.all(base_activations <= 1.0) and np.all(ai_activations <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
