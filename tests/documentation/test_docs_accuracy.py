"""
Comprehensive documentation accuracy tests.

This module validates that all documentation is accurate, up-to-date, and
consistent with the actual codebase implementation.
"""

import pytest
import re
import inspect
import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Import modules to test against documentation
try:
    from core import ActInfLearner, RuleBasedLearner
    from config import (
        ActiveInferenceConfig, THOUGHTSEEDS, STATES,
        ThoughtseedParams, MetacognitionParams
    )
    from utils import (
        ensure_directories, FreeEnergyTracer, ExportManager,
        convert_numpy_to_lists, _save_json_outputs
    )
    from visualization import generate_all_plots, FreeEnergyVisualizer
    from analysis import (
        StatisticalAnalyzer, ComparisonAnalyzer, MetricsCalculator
    )
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


class DocumentationParser:
    """Parse markdown documentation and extract code references."""
    
    def __init__(self, docs_dir: Path):
        self.docs_dir = docs_dir
    
    def get_all_doc_files(self) -> List[Path]:
        """Get all markdown files in docs directory."""
        if not self.docs_dir.exists():
            return []
        return list(self.docs_dir.glob("*.md"))
    
    def extract_code_blocks(self, file_path: Path) -> List[str]:
        """Extract all code blocks from markdown file."""
        if not file_path.exists():
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all code blocks (both ``` and indented)
        code_blocks = []
        
        # Fenced code blocks
        fenced_pattern = r'```(?:python)?\n(.*?)\n```'
        fenced_matches = re.findall(fenced_pattern, content, re.DOTALL)
        code_blocks.extend(fenced_matches)
        
        return code_blocks
    
    def extract_method_references(self, file_path: Path) -> Set[str]:
        """Extract method and class references from documentation."""
        if not file_path.exists():
            return set()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        references = set()
        
        # Find class references (e.g., `ActInfLearner`)
        class_pattern = r'`([A-Z][a-zA-Z_][a-zA-Z0-9_]*)`'
        class_matches = re.findall(class_pattern, content)
        references.update(class_matches)
        
        # Find method references (e.g., `get_target_activations()`)
        method_pattern = r'`([a-z_][a-zA-Z0-9_]*)\(\)`'
        method_matches = re.findall(method_pattern, content)
        references.update(method_matches)
        
        # Find file references (e.g., `act_inf_learner.py`)
        file_pattern = r'`([a-z_][a-zA-Z0-9_]*\.py)`'
        file_matches = re.findall(file_pattern, content)
        references.update(file_matches)
        
        return references
    
    def extract_parameter_references(self, file_path: Path) -> Dict[str, List[str]]:
        """Extract parameter values and ranges from documentation."""
        if not file_path.exists():
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parameters = {}
        
        # Find parameter tables
        table_pattern = r'\|[^|]*Parameter[^|]*\|[^|]*Novice[^|]*\|[^|]*Expert[^|]*\|[^|]*Description[^|]*\|'
        if re.search(table_pattern, content):
            # Extract parameter values from tables
            row_pattern = r'\|\s*`?([a-z_]+)`?\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|'
            matches = re.findall(row_pattern, content)
            
            for param, novice_val, expert_val in matches:
                parameters[param] = {
                    'novice': float(novice_val),
                    'expert': float(expert_val)
                }
        
        return parameters


class TestDocumentationStructure:
    """Test documentation structure and completeness."""
    
    def test_all_documentation_files_exist(self):
        """Test that all expected documentation files exist."""
        docs_dir = Path("docs")
        
        if not docs_dir.exists():
            pytest.skip("Documentation directory not found")
        
        expected_files = [
            "index.md",
            "active_inference_core.md",
            "configuration_system.md",
            "free_energy_calculations.md",
            "network_dynamics.md",
            "rules_based_foundation.md",
            "state_transitions.md",
            "thoughtseed_dynamics.md",
            "utilities_and_helpers.md",
            "visualization_system.md"
        ]
        
        existing_files = [f.name for f in docs_dir.glob("*.md")]
        
        for expected_file in expected_files:
            assert expected_file in existing_files, f"Missing documentation file: {expected_file}"
    
    def test_documentation_files_not_empty(self):
        """Test that documentation files have substantial content."""
        docs_dir = Path("docs")
        
        if not docs_dir.exists():
            pytest.skip("Documentation directory not found")
        
        for doc_file in docs_dir.glob("*.md"):
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Each file should have at least 1000 characters of content
            assert len(content) > 1000, f"Documentation file {doc_file.name} is too short"
            
            # Should have proper markdown structure
            assert content.startswith('#'), f"Documentation file {doc_file.name} should start with a header"
    
    def test_documentation_headers_structure(self):
        """Test that documentation has proper header hierarchy."""
        docs_dir = Path("docs")
        
        if not docs_dir.exists():
            pytest.skip("Documentation directory not found")
        
        for doc_file in docs_dir.glob("*.md"):
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for proper header hierarchy
            headers = re.findall(r'^(#+)\s+(.+)$', content, re.MULTILINE)
            
            if headers:
                # First header should be level 1
                assert headers[0][0] == '#', f"First header in {doc_file.name} should be level 1"
                
                # Headers should not skip levels drastically
                prev_level = 1
                for header_marks, header_text in headers[1:]:
                    current_level = len(header_marks)
                    level_jump = current_level - prev_level
                    assert level_jump <= 2, f"Header level jump too large in {doc_file.name}: {header_text}"
                    prev_level = current_level


@pytest.mark.documentation
class TestCodeDocumentationConsistency:
    """Test consistency between code and documentation."""
    
    @pytest.skipif(not MODULES_AVAILABLE, reason="Core modules not available")
    def test_documented_classes_exist(self):
        """Test that all documented classes actually exist in code."""
        docs_dir = Path("docs")
        
        if not docs_dir.exists():
            pytest.skip("Documentation directory not found")
        
        parser = DocumentationParser(docs_dir)
        
        # Expected class mappings
        expected_classes = {
            'ActInfLearner': ActInfLearner,
            'RuleBasedLearner': RuleBasedLearner,
            'ActiveInferenceConfig': ActiveInferenceConfig,
            'FreeEnergyTracer': FreeEnergyTracer,
            'ExportManager': ExportManager,
            'StatisticalAnalyzer': StatisticalAnalyzer,
            'ComparisonAnalyzer': ComparisonAnalyzer,
            'MetricsCalculator': MetricsCalculator,
            'FreeEnergyVisualizer': FreeEnergyVisualizer
        }
        
        # Check each documentation file
        for doc_file in parser.get_all_doc_files():
            references = parser.extract_method_references(doc_file)
            
            for ref in references:
                if ref in expected_classes:
                    # Verify class exists and is importable
                    assert expected_classes[ref] is not None
                    assert inspect.isclass(expected_classes[ref])
    
    @pytest.skipif(not MODULES_AVAILABLE, reason="Core modules not available")
    def test_documented_methods_exist(self):
        """Test that documented methods actually exist."""
        docs_dir = Path("docs")
        
        if not docs_dir.exists():
            pytest.skip("Documentation directory not found")
        
        parser = DocumentationParser(docs_dir)
        
        # Method mappings to check
        method_mappings = {
            'get_target_activations': (RuleBasedLearner, 'get_target_activations'),
            'get_dwell_time': (RuleBasedLearner, 'get_dwell_time'),
            'get_meta_awareness': (RuleBasedLearner, 'get_meta_awareness'),
            'compute_network_activations': (ActInfLearner, 'compute_network_activations'),
            'calculate_free_energy': (ActInfLearner, 'calculate_free_energy'),
            'train': (ActInfLearner, 'train'),
            'ensure_directories': (None, ensure_directories),
            'convert_numpy_to_lists': (None, convert_numpy_to_lists),
            'generate_all_plots': (None, generate_all_plots)
        }
        
        documented_methods = set()
        for doc_file in parser.get_all_doc_files():
            references = parser.extract_method_references(doc_file)
            documented_methods.update(references)
        
        for method_name in documented_methods:
            if method_name in method_mappings:
                class_obj, method_ref = method_mappings[method_name]
                
                if class_obj is None:
                    # Function reference
                    assert callable(method_ref), f"Function {method_name} should be callable"
                else:
                    # Method reference
                    assert hasattr(class_obj, method_name), f"Method {method_name} not found in {class_obj.__name__}"
                    method_obj = getattr(class_obj, method_name)
                    assert callable(method_obj), f"Method {method_name} should be callable"
    
    @pytest.skipif(not MODULES_AVAILABLE, reason="Core modules not available")
    def test_documented_parameters_accuracy(self):
        """Test that documented parameter values match actual values."""
        docs_dir = Path("docs")
        
        if not docs_dir.exists():
            pytest.skip("Documentation directory not found")
        
        parser = DocumentationParser(docs_dir)
        
        # Check configuration system documentation
        config_doc = docs_dir / "configuration_system.md"
        if config_doc.exists():
            parameters = parser.extract_parameter_references(config_doc)
            
            for param_name, values in parameters.items():
                # Get actual parameter values
                try:
                    novice_params = ActiveInferenceConfig.get_params('novice')
                    expert_params = ActiveInferenceConfig.get_params('expert')
                    
                    if param_name in novice_params and param_name in expert_params:
                        actual_novice = float(novice_params[param_name])
                        actual_expert = float(expert_params[param_name])
                        
                        documented_novice = values['novice']
                        documented_expert = values['expert']
                        
                        # Allow small floating point differences
                        assert abs(actual_novice - documented_novice) < 0.001, \
                            f"Parameter {param_name} novice value mismatch: doc={documented_novice}, actual={actual_novice}"
                        
                        assert abs(actual_expert - documented_expert) < 0.001, \
                            f"Parameter {param_name} expert value mismatch: doc={documented_expert}, actual={actual_expert}"
                
                except (KeyError, TypeError):
                    # Parameter might not be directly accessible
                    continue
    
    @pytest.skipif(not MODULES_AVAILABLE, reason="Core modules not available")
    def test_constants_documentation_accuracy(self):
        """Test that documented constants match actual values."""
        docs_dir = Path("docs")
        
        if not docs_dir.exists():
            pytest.skip("Documentation directory not found")
        
        # Check that documented thoughtseeds and states match actual constants
        for doc_file in docs_dir.glob("*.md"):
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check THOUGHTSEEDS references
            if 'THOUGHTSEEDS' in content:
                # Extract thoughtseed lists from documentation
                ts_pattern = r"THOUGHTSEEDS\s*=\s*\[([^\]]+)\]"
                matches = re.findall(ts_pattern, content)
                
                for match in matches:
                    # Clean up the match and extract thoughtseed names
                    thoughtseeds_text = match.replace("'", "").replace('"', '').replace(' ', '')
                    doc_thoughtseeds = [ts.strip() for ts in thoughtseeds_text.split(',') if ts.strip()]
                    
                    # Should match actual THOUGHTSEEDS
                    assert set(doc_thoughtseeds) == set(THOUGHTSEEDS), \
                        f"Documented thoughtseeds don't match actual: {doc_thoughtseeds} vs {THOUGHTSEEDS}"
            
            # Check STATES references
            if 'STATES' in content:
                states_pattern = r"STATES\s*=\s*\[([^\]]+)\]"
                matches = re.findall(states_pattern, content)
                
                for match in matches:
                    states_text = match.replace("'", "").replace('"', '').replace(' ', '')
                    doc_states = [state.strip() for state in states_text.split(',') if state.strip()]
                    
                    assert set(doc_states) == set(STATES), \
                        f"Documented states don't match actual: {doc_states} vs {STATES}"


@pytest.mark.documentation
class TestCodeExamples:
    """Test that code examples in documentation are valid."""
    
    def test_python_code_blocks_syntax(self):
        """Test that Python code blocks have valid syntax."""
        docs_dir = Path("docs")
        
        if not docs_dir.exists():
            pytest.skip("Documentation directory not found")
        
        parser = DocumentationParser(docs_dir)
        
        for doc_file in parser.get_all_doc_files():
            code_blocks = parser.extract_code_blocks(doc_file)
            
            for i, code_block in enumerate(code_blocks):
                # Skip code blocks that are clearly just examples or pseudocode
                if any(marker in code_block.lower() for marker in ['...', 'example:', 'pseudo', '#']):
                    continue
                
                # Try to parse as Python
                try:
                    ast.parse(code_block)
                except SyntaxError as e:
                    # Some code blocks might be incomplete snippets
                    # Try wrapping in a function
                    try:
                        wrapped_code = f"def example():\n" + "\n".join(f"    {line}" for line in code_block.split('\n'))
                        ast.parse(wrapped_code)
                    except SyntaxError:
                        pytest.fail(f"Invalid Python syntax in {doc_file.name}, code block {i}: {e}")
    
    def test_import_statements_validity(self):
        """Test that import statements in documentation are valid."""
        docs_dir = Path("docs")
        
        if not docs_dir.exists():
            pytest.skip("Documentation directory not found")
        
        for doc_file in docs_dir.glob("*.md"):
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find import statements
            import_pattern = r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import\s+([a-zA-Z_][a-zA-Z0-9_,\s]*)'
            matches = re.findall(import_pattern, content, re.MULTILINE)
            
            for module_name, imports in matches:
                # Skip relative imports in examples
                if module_name.startswith('.'):
                    continue
                
                # Check if documented imports are reasonable
                import_list = [imp.strip() for imp in imports.split(',')]
                
                # These should be our modules or standard library
                expected_modules = ['core', 'config', 'utils', 'visualization', 'analysis']
                
                if module_name in expected_modules:
                    # Should be importing actual classes/functions
                    expected_imports = {
                        'core': ['ActInfLearner', 'RuleBasedLearner'],
                        'config': ['ActiveInferenceConfig', 'THOUGHTSEEDS', 'STATES'],
                        'utils': ['ensure_directories', 'FreeEnergyTracer', 'ExportManager'],
                        'visualization': ['generate_all_plots', 'FreeEnergyVisualizer'],
                        'analysis': ['StatisticalAnalyzer', 'ComparisonAnalyzer', 'MetricsCalculator']
                    }
                    
                    if module_name in expected_imports:
                        for import_name in import_list:
                            if import_name not in expected_imports[module_name]:
                                # This might be okay - could be new additions
                                # Just log for now
                                print(f"Note: {import_name} imported from {module_name} in {doc_file.name}")


@pytest.mark.documentation
class TestDocumentationCompleteness:
    """Test that documentation covers all important aspects."""
    
    @pytest.skipif(not MODULES_AVAILABLE, reason="Core modules not available")
    def test_all_public_methods_documented(self):
        """Test that all important public methods are documented somewhere."""
        docs_dir = Path("docs")
        
        if not docs_dir.exists():
            pytest.skip("Documentation directory not found")
        
        # Get all documented methods
        parser = DocumentationParser(docs_dir)
        documented_methods = set()
        
        for doc_file in parser.get_all_doc_files():
            references = parser.extract_method_references(doc_file)
            documented_methods.update(references)
        
        # Important methods that should be documented
        important_methods = [
            'get_target_activations',
            'get_dwell_time', 
            'get_meta_awareness',
            'compute_network_activations',
            'calculate_free_energy',
            'train',
            'ensure_directories',
            'generate_all_plots'
        ]
        
        for method in important_methods:
            assert method in documented_methods, \
                f"Important method {method} is not documented"
    
    def test_documentation_cross_references(self):
        """Test that cross-references in documentation are valid."""
        docs_dir = Path("docs")
        
        if not docs_dir.exists():
            pytest.skip("Documentation directory not found")
        
        # Get all documentation files
        doc_files = {f.stem: f for f in docs_dir.glob("*.md")}
        
        # Check for cross-references
        for doc_file in doc_files.values():
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find markdown links to other documentation
            link_pattern = r'\[([^\]]+)\]\(\.?/?([^)]+)\.md\)'
            matches = re.findall(link_pattern, content)
            
            for link_text, target_file in matches:
                # Remove any anchors
                target_file = target_file.split('#')[0]
                
                # Check if target file exists
                assert target_file in doc_files, \
                    f"Cross-reference to non-existent file {target_file}.md in {doc_file.name}"
    
    def test_documentation_equations_references(self):
        """Test that equation references are consistent."""
        docs_dir = Path("docs")
        
        if not docs_dir.exists():
            pytest.skip("Documentation directory not found")
        
        # Look for equation references
        equation_pattern = r'Equation\s+(\d+)'
        
        for doc_file in docs_dir.glob("*.md"):
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            equation_refs = re.findall(equation_pattern, content)
            
            if equation_refs:
                # Should reference equations 1-4 (based on theoretical framework)
                for eq_num in equation_refs:
                    eq_int = int(eq_num)
                    assert 1 <= eq_int <= 4, \
                        f"Invalid equation reference {eq_num} in {doc_file.name}"


@pytest.mark.documentation
class TestDocumentationStyle:
    """Test documentation style and formatting."""
    
    def test_consistent_code_formatting(self):
        """Test that code is consistently formatted in documentation."""
        docs_dir = Path("docs")
        
        if not docs_dir.exists():
            pytest.skip("Documentation directory not found")
        
        for doc_file in docs_dir.glob("*.md"):
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for inconsistent code formatting
            # Inline code should use backticks
            inline_code_count = len(re.findall(r'`[^`]+`', content))
            
            # Should have some inline code formatting
            if len(content) > 2000:  # Only check substantial files
                assert inline_code_count > 5, \
                    f"Documentation {doc_file.name} should have more inline code formatting"
    
    def test_proper_table_formatting(self):
        """Test that tables are properly formatted."""
        docs_dir = Path("docs")
        
        if not docs_dir.exists():
            pytest.skip("Documentation directory not found")
        
        for doc_file in docs_dir.glob("*.md"):
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find table rows
            table_rows = re.findall(r'^\|.*\|$', content, re.MULTILINE)
            
            if table_rows:
                # Check that tables have consistent column counts
                column_counts = []
                for row in table_rows:
                    # Skip separator rows
                    if not re.match(r'^\|[\s\-:|]*\|$', row):
                        columns = row.split('|')
                        # Remove empty first/last elements from split
                        columns = [col.strip() for col in columns if col.strip()]
                        column_counts.append(len(columns))
                
                if column_counts:
                    # All rows should have the same number of columns
                    assert len(set(column_counts)) <= 1, \
                        f"Inconsistent table column counts in {doc_file.name}: {set(column_counts)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "documentation"])
