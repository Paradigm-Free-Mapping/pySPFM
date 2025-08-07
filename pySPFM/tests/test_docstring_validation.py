"""Test module for validating docstring changes in test functions."""
import inspect

from pySPFM.tests.test_lars import test_solve_regularization_path


def test_solve_regularization_path_docstring_exists():
    """Test that test_solve_regularization_path has a docstring."""
    assert test_solve_regularization_path.__doc__ is not None
    assert len(test_solve_regularization_path.__doc__.strip()) > 0


def test_solve_regularization_path_docstring_content():
    """Test that test_solve_regularization_path docstring contains expected content."""
    docstring = test_solve_regularization_path.__doc__
    
    # Check that it contains the main description
    assert "Test the solve_regularization_path function." in docstring
    
    # Check that it contains the Parameters section
    assert "Parameters" in docstring
    assert "----------" in docstring
    
    # Check that all three parameters are documented
    assert "sim_data : str" in docstring
    assert "sim_hrf : str" in docstring  
    assert "coef_path_results : str" in docstring
    
    # Check parameter descriptions
    assert "Path to the simulated data." in docstring
    assert "Path to the simulated HRF." in docstring
    assert "Path to the coefficient path results." in docstring


def test_solve_regularization_path_docstring_format():
    """Test that test_solve_regularization_path docstring follows NumPy format."""
    docstring = test_solve_regularization_path.__doc__
    
    # Check for proper NumPy-style formatting
    lines = docstring.split('\n')
    
    # Should have proper indentation structure
    assert any("Parameters" in line for line in lines)
    assert any("----------" in line for line in lines)
    
    # Check that parameter descriptions are properly indented
    param_lines = [line for line in lines if " : " in line and ("sim_data" in line or "sim_hrf" in line or "coef_path_results" in line)]
    assert len(param_lines) == 3
    
    # Each parameter line should be properly indented
    for line in param_lines:
        assert line.startswith("    ")  # 4 spaces indentation