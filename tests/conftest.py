"""
Pytest configuration file for the meditation simulation framework.

This file provides fixtures and configuration for all tests including:
- Mock data generation
- Test learner instances
- File system management
- Documentation parsing utilities
"""

import pytest
import numpy as np
import tempfile
import shutil
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Import all modules to test
try:
    from core import ActInfLearner, RuleBasedLearner
    from config import ActiveInferenceConfig, THOUGHTSEEDS, STATES
    from utils import (
        ensure_directories, FreeEnergyTracer, ExportManager, ExportConfig,
        convert_numpy_to_lists, _save_json_outputs
    )
    from visualization import generate_all_plots, FreeEnergyVisualizer
    from analysis import (
        StatisticalAnalyzer, ComparisonAnalyzer, MetricsCalculator,
        NetworkAnalyzer, TimeSeriesAnalyzer
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Import error in conftest.py: {e}")
    IMPORT_SUCCESS = False

# Test constants
TEST_THOUGHTSEEDS = ['breath_focus', 'pain_discomfort', 'pending_tasks', 
                    'self_reflection', 'equanimity']
TEST_STATES = ['breath_control', 'mind_wandering', 'meta_awareness', 'redirect_breath']
TEST_NETWORKS = ['DMN', 'VAN', 'DAN', 'FPN']


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_learner_data():
    """Generate mock data for learner testing."""
    timesteps = 50
    
    return {
        'timesteps': timesteps,
        'activations_history': [
            np.random.rand(len(TEST_THOUGHTSEEDS)) for _ in range(timesteps)
        ],
        'network_activations_history': [
            {net: np.random.rand() for net in TEST_NETWORKS} for _ in range(timesteps)
        ],
        'state_history': [
            np.random.choice(TEST_STATES) for _ in range(timesteps)
        ],
        'meta_awareness_history': np.random.rand(timesteps).tolist(),
        'free_energy_history': np.random.rand(timesteps).tolist(),
        'prediction_error_history': np.random.rand(timesteps).tolist(),
        'precision_history': np.random.rand(timesteps).tolist(),
        'dominant_ts_history': [
            np.random.choice(TEST_THOUGHTSEEDS) for _ in range(timesteps)
        ]
    }


@pytest.fixture
def novice_learner():
    """Create a novice learner instance for testing."""
    if not IMPORT_SUCCESS:
        pytest.skip("Failed to import required modules")
    
    learner = ActInfLearner(experience_level='novice', timesteps_per_cycle=20)
    return learner


@pytest.fixture
def expert_learner():
    """Create an expert learner instance for testing."""
    if not IMPORT_SUCCESS:
        pytest.skip("Failed to import required modules")
    
    learner = ActInfLearner(experience_level='expert', timesteps_per_cycle=20)
    return learner


@pytest.fixture
def trained_novice_learner(temp_dir):
    """Create a trained novice learner with realistic data."""
    if not IMPORT_SUCCESS:
        pytest.skip("Failed to import required modules")
    
    # Change to temp directory for clean testing
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    try:
        learner = ActInfLearner(experience_level='novice', timesteps_per_cycle=30)
        learner.train()
        yield learner
    finally:
        os.chdir(original_dir)


@pytest.fixture
def trained_expert_learner(temp_dir):
    """Create a trained expert learner with realistic data."""
    if not IMPORT_SUCCESS:
        pytest.skip("Failed to import required modules")
    
    # Change to temp directory for clean testing
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    try:
        learner = ActInfLearner(experience_level='expert', timesteps_per_cycle=30)
        learner.train()
        yield learner
    finally:
        os.chdir(original_dir)


@pytest.fixture
def mock_network_activations():
    """Generate mock network activations."""
    return {
        'DMN': 0.45,
        'VAN': 0.62,
        'DAN': 0.38,
        'FPN': 0.71
    }


@pytest.fixture
def mock_thoughtseed_activations():
    """Generate mock thoughtseed activations."""
    return np.array([0.7, 0.2, 0.1, 0.4, 0.6])  # breath_focus, pain, tasks, reflection, equanimity


@pytest.fixture
def free_energy_tracer(temp_dir):
    """Create a FreeEnergyTracer instance for testing."""
    if not IMPORT_SUCCESS:
        pytest.skip("Failed to import required modules")
    
    return FreeEnergyTracer(output_dir=os.path.join(temp_dir, "fe_traces"))


@pytest.fixture
def export_config():
    """Create export configuration for testing."""
    if not IMPORT_SUCCESS:
        pytest.skip("Failed to import required modules")
    
    return ExportConfig(
        formats=['json', 'csv'],
        include_metadata=True,
        include_time_series=True,
        include_analysis=True,
        statistical_summaries=True
    )


@pytest.fixture
def sample_json_data(temp_dir):
    """Create sample JSON data files for testing visualization."""
    data_dir = os.path.join(temp_dir, "results_act_inf", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Sample thoughtseed data
    thoughtseed_data = {
        "time_series": {
            "activations_history": [[0.7, 0.2, 0.1, 0.4, 0.6] for _ in range(20)],
            "network_activations_history": [
                {"DMN": 0.4, "VAN": 0.6, "DAN": 0.5, "FPN": 0.7} for _ in range(20)
            ],
            "state_history": ["breath_control"] * 10 + ["mind_wandering"] * 10,
            "meta_awareness_history": [0.5] * 20,
            "free_energy_history": [1.2] * 20,
            "dominant_ts_history": ["breath_focus"] * 20
        },
        "activation_means_by_state": {
            "breath_control": {"breath_focus": 0.7, "pain_discomfort": 0.2}
        }
    }
    
    # Sample active inference data
    ai_data = {
        "precision_weight": 0.4,
        "complexity_penalty": 0.4,
        "learning_rate": 0.01,
        "average_free_energy_by_state": {
            "breath_control": 1.2,
            "mind_wandering": 1.8
        }
    }
    
    # Save sample data
    for level in ['novice', 'expert']:
        with open(os.path.join(data_dir, f"thoughtseed_params_{level}.json"), 'w') as f:
            json.dump(thoughtseed_data, f)
        
        with open(os.path.join(data_dir, f"active_inference_params_{level}.json"), 'w') as f:
            json.dump(ai_data, f)
    
    return temp_dir


@pytest.fixture
def documentation_files():
    """Get list of all documentation files to test."""
    docs_dir = Path("docs")
    
    if docs_dir.exists():
        return list(docs_dir.glob("*.md"))
    else:
        return []


class MockLearnerForTesting:
    """Mock learner class for testing purposes."""
    
    def __init__(self, experience_level='novice'):
        self.experience_level = experience_level
        self.thoughtseeds = TEST_THOUGHTSEEDS
        self.states = TEST_STATES
        self.networks = TEST_NETWORKS
        self.num_thoughtseeds = len(TEST_THOUGHTSEEDS)
        self.timesteps = 30
        
        # Generate realistic mock data
        self.activations_history = [
            np.random.rand(self.num_thoughtseeds) for _ in range(self.timesteps)
        ]
        self.network_activations_history = [
            {net: np.random.rand() for net in TEST_NETWORKS} 
            for _ in range(self.timesteps)
        ]
        self.state_history = [
            np.random.choice(TEST_STATES) for _ in range(self.timesteps)
        ]
        self.meta_awareness_history = np.random.rand(self.timesteps).tolist()
        self.free_energy_history = np.random.rand(self.timesteps).tolist()
        self.prediction_error_history = np.random.rand(self.timesteps).tolist()
        self.precision_history = np.random.rand(self.timesteps).tolist()
        self.dominant_ts_history = [
            np.random.choice(TEST_THOUGHTSEEDS) for _ in range(self.timesteps)
        ]
        
        # Mock parameters
        self.precision_weight = 0.4 if experience_level == 'novice' else 0.5
        self.complexity_penalty = 0.4 if experience_level == 'novice' else 0.2
        self.learning_rate = 0.01 if experience_level == 'novice' else 0.02
        
        # Mock learned network profiles (needed for data management)
        from config import NETWORK_PROFILES
        self.learned_network_profiles = {
            "thoughtseed_contributions": {
                ts: NETWORK_PROFILES["thoughtseed_contributions"][ts] for ts in self.thoughtseeds
            },
            "state_network_expectations": {
                state: {
                    exp_level: NETWORK_PROFILES["state_expected_profiles"][state][exp_level]
                    for exp_level in ['novice', 'expert']
                } for state in self.states
            }
        }


@pytest.fixture
def mock_learner():
    """Create a mock learner instance."""
    return MockLearnerForTesting()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "documentation: marks tests as documentation validation")
    config.addinivalue_line("markers", "visualization: marks tests that require plotting")
    config.addinivalue_line("markers", "export: marks tests that test data export functionality")


def pytest_collection_modifyitems(config, items):
    """Automatically mark certain test types."""
    for item in items:
        # Mark slow tests
        if "train" in item.name or "simulation" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if item.fspath.basename.startswith("test_integration"):
            item.add_marker(pytest.mark.integration)
        
        # Mark documentation tests
        if "documentation" in str(item.fspath) or "docs" in item.name:
            item.add_marker(pytest.mark.documentation)
        
        # Mark visualization tests
        if "plot" in item.name or "visual" in item.name:
            item.add_marker(pytest.mark.visualization)
        
        # Mark export tests
        if "export" in item.name or "save" in item.name:
            item.add_marker(pytest.mark.export)
