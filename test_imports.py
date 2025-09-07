#!/usr/bin/env python3
"""
Simple import test script to validate all modules.
"""

import sys

try:
    # Test core imports
    from core import ActInfLearner, RuleBasedLearner
    print("✅ Core modules imported successfully")
    
    # Test config imports
    from config import ActiveInferenceConfig, THOUGHTSEEDS, STATES
    print("✅ Configuration modules imported successfully")
    
    # Test utils imports
    from utils import ensure_directories, FreeEnergyTracer, ExportManager
    print("✅ Utility modules imported successfully")
    
    # Test visualization imports
    from visualization import generate_all_plots, FreeEnergyVisualizer
    print("✅ Visualization modules imported successfully")
    
    # Test analysis imports
    from analysis import StatisticalAnalyzer, ComparisonAnalyzer, MetricsCalculator
    print("✅ Analysis modules imported successfully")
    
    print("\n🎉 All modules imported successfully!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"💥 Unexpected error: {e}")
    sys.exit(1)
