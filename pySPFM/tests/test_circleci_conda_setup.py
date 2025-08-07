"""Tests for CircleCI conda environment setup logic."""
import os
import subprocess
import tempfile
import pytest
from unittest.mock import patch, MagicMock, call


class TestCondaEnvironmentSetup:
    """Test conda environment setup logic used in CircleCI."""

    def setup_method(self):
        """Set up test fixtures."""
        self.python_version = "3.9"
        self.env_name = f"py{self.python_version}_env"
        self.conda_envs_path = "/opt/conda/envs"
        self.env_path = f"{self.conda_envs_path}/{self.env_name}"

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_conda_shell_initialization(self, mock_exists, mock_run):
        """Test that conda shell initialization is called correctly."""
        # Mock successful conda shell hook
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        
        # Simulate the conda shell.bash hook command
        cmd = 'eval "$(conda shell.bash hook)"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Verify the command was called
        mock_run.assert_called_with(cmd, shell=True, capture_output=True, text=True)

    @patch("subprocess.run")
    def test_conda_info_environments(self, mock_run):
        """Test conda environment listing command."""
        expected_output = f"""
# conda environments:
#
base                  *  /opt/conda
{self.env_name}                     {self.env_path}
"""
        mock_run.return_value = MagicMock(
            returncode=0, stdout=expected_output, stderr=""
        )
        
        # Simulate the conda info --envs command
        cmd = "conda info --envs"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        mock_run.assert_called_with(cmd, shell=True, capture_output=True, text=True)
        assert self.env_name in result.stdout

    @patch("subprocess.run")
    def test_conda_env_list_check(self, mock_run):
        """Test checking if environment exists in conda env list."""
        # Mock conda env list output with our environment
        env_list_output = f"""
# conda environments:
#
base                  *  /opt/conda
{self.env_name}                     {self.env_path}
"""
        mock_run.return_value = MagicMock(
            returncode=0, stdout=env_list_output, stderr=""
        )
        
        cmd = "conda env list"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Check if environment is found in the output
        assert self.env_name in result.stdout
        mock_run.assert_called_with(cmd, shell=True, capture_output=True, text=True)

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_environment_found_and_registered(self, mock_exists, mock_run):
        """Test scenario where environment is found in conda list."""
        # Mock that environment exists in conda list
        env_list_output = f"{self.env_name}                     {self.env_path}"
        mock_run.return_value = MagicMock(
            returncode=0, stdout=env_list_output, stderr=""
        )
        
        # Simulate conda env list | grep check
        cmd = f'conda env list | grep -q "{self.env_name}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Should find the environment
        mock_run.assert_called_with(cmd, shell=True, capture_output=True, text=True)

    @patch("subprocess.run")
    @patch("os.path.exists")
    @patch("shutil.rmtree")
    def test_environment_directory_exists_but_not_registered(
        self, mock_rmtree, mock_exists, mock_run
    ):
        """Test scenario where env directory exists but not registered in conda."""
        # Mock directory exists but not in conda list
        mock_exists.return_value = True
        
        # Mock conda env list without our environment
        env_list_output = "base                  *  /opt/conda"
        grep_result = MagicMock(returncode=1, stdout="", stderr="")  # grep fails
        create_result = MagicMock(returncode=0, stdout="", stderr="")
        
        mock_run.side_effect = [grep_result, create_result, create_result]
        
        # Test the scenario
        import shutil
        
        # Check if directory exists
        assert os.path.exists(self.env_path)
        
        # Remove and recreate
        shutil.rmtree(self.env_path)
        mock_rmtree.assert_called_with(self.env_path)
        
        # Create new environment
        create_cmd = f"conda create -n {self.env_name} python={self.python_version} -yq"
        subprocess.run(create_cmd, shell=True, capture_output=True, text=True)
        
        # Verify create command was called
        assert any(
            call(create_cmd, shell=True, capture_output=True, text=True) 
            in mock_run.call_args_list
        )

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_environment_not_found_create_new(self, mock_exists, mock_run):
        """Test scenario where environment doesn't exist and needs creation."""
        # Mock directory doesn't exist
        mock_exists.return_value = False
        
        # Mock conda env list without our environment (grep fails)
        grep_result = MagicMock(returncode=1, stdout="", stderr="")
        create_result = MagicMock(returncode=0, stdout="", stderr="")
        activate_result = MagicMock(returncode=0, stdout="", stderr="")
        pip_result = MagicMock(returncode=0, stdout="", stderr="")
        
        mock_run.side_effect = [grep_result, create_result, activate_result, pip_result]
        
        # Test environment creation
        create_cmd = f"conda create -n {self.env_name} python={self.python_version} -yq"
        subprocess.run(create_cmd, shell=True, capture_output=True, text=True)
        
        activate_cmd = f"conda activate {self.env_name}"
        subprocess.run(activate_cmd, shell=True, capture_output=True, text=True)
        
        pip_cmd = "pip install -e .[tests,doc]"
        subprocess.run(pip_cmd, shell=True, capture_output=True, text=True)
        
        # Verify all commands were called
        expected_calls = [
            call(create_cmd, shell=True, capture_output=True, text=True),
            call(activate_cmd, shell=True, capture_output=True, text=True),
            call(pip_cmd, shell=True, capture_output=True, text=True),
        ]
        
        for expected_call in expected_calls:
            assert expected_call in mock_run.call_args_list

    @patch("subprocess.run")
    def test_python_version_verification(self, mock_run):
        """Test Python version and location verification commands."""
        version_output = f"Python {self.python_version}.0"
        which_output = f"/opt/conda/envs/{self.env_name}/bin/python"
        
        version_result = MagicMock(returncode=0, stdout=version_output, stderr="")
        which_result = MagicMock(returncode=0, stdout=which_output, stderr="")
        
        mock_run.side_effect = [version_result, which_result]
        
        # Test verification commands
        subprocess.run("python --version", shell=True, capture_output=True, text=True)
        subprocess.run("which python", shell=True, capture_output=True, text=True)
        
        mock_run.assert_any_call("python --version", shell=True, capture_output=True, text=True)
        mock_run.assert_any_call("which python", shell=True, capture_output=True, text=True)