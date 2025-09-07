#!/usr/bin/env python3
"""
Comprehensive Test Runner for Meditation Simulation Framework

This script runs all tests systematically, providing detailed reporting
and validation of the entire system.

Features:
- Unit tests for all modules
- Integration tests for full system
- Documentation accuracy validation
- Performance testing
- Coverage reporting
- Detailed error analysis
"""

import sys
import subprocess
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


class TestRunner:
    """Comprehensive test runner with detailed reporting."""
    
    def __init__(self, verbose: bool = True, coverage: bool = True):
        self.verbose = verbose
        self.coverage = coverage
        self.results = {}
        self.start_time = time.time()
        
    def run_command(self, cmd: List[str], description: str) -> Tuple[bool, str, str]:
        """Run a command and capture output."""
        if self.verbose:
            print(f"\n🔧 {description}")
            print(f"   Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            
            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr
            
            if self.verbose:
                status = "✅ PASSED" if success else "❌ FAILED"
                print(f"   Status: {status}")
                if not success and stderr:
                    print(f"   Error: {stderr[:200]}...")
            
            return success, stdout, stderr
            
        except subprocess.TimeoutExpired:
            print(f"   ⏰ TIMEOUT: Command took longer than 5 minutes")
            return False, "", "Command timed out"
        except Exception as e:
            print(f"   💥 EXCEPTION: {str(e)}")
            return False, "", str(e)
    
    def run_unit_tests(self) -> bool:
        """Run all unit tests."""
        print("\n" + "="*60)
        print("🧪 RUNNING UNIT TESTS")
        print("="*60)
        
        test_files = [
            ("Core Functionality", "tests/unit/test_core.py"),
            ("Configuration System", "tests/unit/test_config.py"),
            ("Utilities", "tests/unit/test_utils.py"),
            ("Analysis Modules", "tests/unit/test_analysis.py"),
            ("Visualization System", "tests/unit/test_visualization.py")
        ]
        
        unit_results = []
        
        for description, test_file in test_files:
            if os.path.exists(test_file):
                cmd = ["python", "-m", "pytest", test_file, "-v"]
                if self.coverage:
                    cmd.extend(["--cov=core", "--cov=config", "--cov=utils", 
                               "--cov=visualization", "--cov=analysis"])
                
                success, stdout, stderr = self.run_command(cmd, f"Unit Tests: {description}")
                unit_results.append((description, success, stdout, stderr))
            else:
                print(f"   ⚠️  SKIPPED: {test_file} not found")
                unit_results.append((description, None, "", f"File not found: {test_file}"))
        
        self.results['unit_tests'] = unit_results
        
        # Count successes
        passed = sum(1 for _, success, _, _ in unit_results if success is True)
        failed = sum(1 for _, success, _, _ in unit_results if success is False)
        skipped = sum(1 for _, success, _, _ in unit_results if success is None)
        
        print(f"\n📊 Unit Test Summary:")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⚠️  Skipped: {skipped}")
        
        return failed == 0
    
    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        print("\n" + "="*60)
        print("🔗 RUNNING INTEGRATION TESTS")
        print("="*60)
        
        integration_files = [
            ("Full System Integration", "tests/integration/test_full_system.py"),
            ("Enhanced Features", "tests/integration/test_enhanced_features.py")
        ]
        
        integration_results = []
        
        for description, test_file in integration_files:
            if os.path.exists(test_file):
                cmd = ["python", "-m", "pytest", test_file, "-v", "-m", "integration"]
                success, stdout, stderr = self.run_command(cmd, f"Integration: {description}")
                integration_results.append((description, success, stdout, stderr))
            else:
                print(f"   ⚠️  SKIPPED: {test_file} not found")
                integration_results.append((description, None, "", f"File not found: {test_file}"))
        
        self.results['integration_tests'] = integration_results
        
        # Count successes
        passed = sum(1 for _, success, _, _ in integration_results if success is True)
        failed = sum(1 for _, success, _, _ in integration_results if success is False)
        skipped = sum(1 for _, success, _, _ in integration_results if success is None)
        
        print(f"\n📊 Integration Test Summary:")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⚠️  Skipped: {skipped}")
        
        return failed == 0
    
    def run_documentation_tests(self) -> bool:
        """Run documentation accuracy tests."""
        print("\n" + "="*60)
        print("📚 RUNNING DOCUMENTATION TESTS")
        print("="*60)
        
        doc_test_file = "tests/documentation/test_docs_accuracy.py"
        
        if os.path.exists(doc_test_file):
            cmd = ["python", "-m", "pytest", doc_test_file, "-v", "-m", "documentation"]
            success, stdout, stderr = self.run_command(cmd, "Documentation Accuracy Validation")
            
            self.results['documentation_tests'] = [(
                "Documentation Accuracy", success, stdout, stderr
            )]
            
            if success:
                print(f"\n📊 Documentation Test Summary:")
                print(f"   ✅ Documentation is accurate and consistent")
            else:
                print(f"\n📊 Documentation Test Summary:")
                print(f"   ❌ Documentation has inconsistencies")
            
            return success
        else:
            print(f"   ⚠️  SKIPPED: Documentation tests not found")
            self.results['documentation_tests'] = [(
                "Documentation Accuracy", None, "", "File not found"
            )]
            return True  # Don't fail if doc tests missing
    
    def run_enhanced_simulation_test(self) -> bool:
        """Run enhanced simulation to verify full system functionality."""
        print("\n" + "="*60)
        print("🚀 RUNNING ENHANCED SIMULATION TEST")
        print("="*60)
        
        enhanced_script = "enhanced_simulation.py"
        
        if os.path.exists(enhanced_script):
            cmd = ["python", enhanced_script, "--experience", "both", 
                   "--timesteps", "30", "--no-visualizations"]
            success, stdout, stderr = self.run_command(cmd, "Enhanced Simulation Test")
            
            self.results['enhanced_simulation'] = success
            
            if success:
                print(f"\n📊 Enhanced Simulation Summary:")
                print(f"   ✅ Full system integration successful")
                print(f"   🔬 Free energy tracing operational")
                print(f"   📈 Analysis pipeline functional")
                print(f"   📤 Export system working")
            else:
                print(f"\n📊 Enhanced Simulation Summary:")
                print(f"   ❌ System integration issues detected")
                if stderr:
                    print(f"   Error details: {stderr[:300]}")
            
            return success
        else:
            print(f"   ⚠️  SKIPPED: Enhanced simulation script not found")
            self.results['enhanced_simulation'] = None
            return True
    
    def run_quick_import_test(self) -> bool:
        """Test that all modules can be imported successfully."""
        print("\n" + "="*60)
        print("📦 RUNNING IMPORT TESTS")
        print("="*60)
        
        import_test_script = """
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
    
    print("\\n🎉 All modules imported successfully!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"💥 Unexpected error: {e}")
    sys.exit(1)
"""
        
        cmd = ["python", "-c", import_test_script]
        success, stdout, stderr = self.run_command(cmd, "Module Import Validation")
        
        self.results['import_tests'] = success
        
        if success:
            print(f"\n📊 Import Test Summary:")
            print(f"   ✅ All modules importable")
        else:
            print(f"\n📊 Import Test Summary:")
            print(f"   ❌ Import issues detected")
        
        return success
    
    def generate_coverage_report(self) -> bool:
        """Generate comprehensive coverage report."""
        if not self.coverage:
            return True
        
        print("\n" + "="*60)
        print("📊 GENERATING COVERAGE REPORT")
        print("="*60)
        
        # Run all tests with coverage
        cmd = ["python", "-m", "pytest", 
               "tests/unit/", "tests/integration/", 
               "--cov=core", "--cov=config", "--cov=utils", 
               "--cov=visualization", "--cov=analysis",
               "--cov-report=term-missing", 
               "--cov-report=html:htmlcov",
               "--cov-fail-under=70"]
        
        success, stdout, stderr = self.run_command(cmd, "Coverage Analysis")
        
        if success:
            print(f"\n📊 Coverage Report:")
            print(f"   ✅ Coverage report generated")
            print(f"   📁 HTML report: htmlcov/index.html")
        else:
            print(f"\n📊 Coverage Report:")
            print(f"   ⚠️  Coverage below threshold or other issues")
        
        self.results['coverage'] = success
        return success
    
    def run_performance_tests(self) -> bool:
        """Run performance tests."""
        print("\n" + "="*60)
        print("⚡ RUNNING PERFORMANCE TESTS")
        print("="*60)
        
        # Run performance-related tests
        cmd = ["python", "-m", "pytest", 
               "tests/", "-v", "-m", "slow",
               "--durations=10"]
        
        success, stdout, stderr = self.run_command(cmd, "Performance Testing")
        
        self.results['performance_tests'] = success
        
        if success:
            print(f"\n📊 Performance Test Summary:")
            print(f"   ✅ Performance within acceptable limits")
        else:
            print(f"\n📊 Performance Test Summary:")
            print(f"   ⚠️  Performance issues or test failures detected")
        
        return success
    
    def generate_final_report(self):
        """Generate comprehensive final report."""
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        total_time = time.time() - self.start_time
        
        print(f"\n⏱️  Total Test Duration: {total_time:.2f} seconds")
        
        # Summary of all test categories
        categories = [
            ("Import Tests", self.results.get('import_tests')),
            ("Unit Tests", all(success for _, success, _, _ in self.results.get('unit_tests', []) if success is not None)),
            ("Integration Tests", all(success for _, success, _, _ in self.results.get('integration_tests', []) if success is not None)),
            ("Documentation Tests", all(success for _, success, _, _ in self.results.get('documentation_tests', []) if success is not None)),
            ("Enhanced Simulation", self.results.get('enhanced_simulation')),
            ("Coverage Analysis", self.results.get('coverage')),
            ("Performance Tests", self.results.get('performance_tests'))
        ]
        
        print(f"\n📊 Test Category Summary:")
        overall_success = True
        
        for category, success in categories:
            if success is True:
                print(f"   ✅ {category}: PASSED")
            elif success is False:
                print(f"   ❌ {category}: FAILED")
                overall_success = False
            else:
                print(f"   ⚠️  {category}: SKIPPED")
        
        print(f"\n🎯 Overall Result:")
        if overall_success:
            print(f"   🎉 ALL TESTS PASSED - System is fully functional!")
            print(f"   ✨ Ready for production use")
        else:
            print(f"   ⚠️  SOME TESTS FAILED - Issues need attention")
            print(f"   🔧 Review failed tests and fix issues")
        
        # Detailed failure analysis
        if not overall_success:
            print(f"\n🔍 Failure Analysis:")
            
            for category_name, category_results in self.results.items():
                if category_name in ['unit_tests', 'integration_tests', 'documentation_tests']:
                    for test_name, success, stdout, stderr in category_results:
                        if success is False:
                            print(f"   ❌ {category_name} - {test_name}")
                            if stderr:
                                print(f"      Error: {stderr[:150]}...")
        
        return overall_success


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Test Runner for Meditation Simulation Framework"
    )
    
    parser.add_argument(
        '--no-coverage', 
        action='store_true',
        help='Skip coverage analysis'
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run only essential tests (imports and unit tests)'
    )
    
    parser.add_argument(
        '--performance-only', 
        action='store_true',
        help='Run only performance tests'
    )
    
    parser.add_argument(
        '--docs-only', 
        action='store_true',
        help='Run only documentation tests'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        default=True,
        help='Verbose output (default: True)'
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner(
        verbose=args.verbose,
        coverage=not args.no_coverage
    )
    
    print("🚀 MEDITATION SIMULATION FRAMEWORK - COMPREHENSIVE TESTING")
    print("="*80)
    print(f"📅 Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Coverage: {'Enabled' if runner.coverage else 'Disabled'}")
    
    success = True
    
    if args.docs_only:
        success = runner.run_documentation_tests()
    elif args.performance_only:
        success = runner.run_performance_tests()
    elif args.quick:
        success = (runner.run_quick_import_test() and 
                  runner.run_unit_tests())
    else:
        # Full test suite
        success = (
            runner.run_quick_import_test() and
            runner.run_unit_tests() and
            runner.run_integration_tests() and
            runner.run_documentation_tests() and
            runner.run_enhanced_simulation_test()
        )
        
        if runner.coverage:
            runner.generate_coverage_report()
        
        runner.run_performance_tests()
    
    # Generate final report
    runner.generate_final_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
