import pytest
import sys
import os

def test_imports():
    """Verify that core modules can be imported without syntax errors."""
    import utils
    import models
    import training
    import evaluation
    assert True

def test_config_paths():
    """Ensure standard project directories exist."""
    assert os.path.exists("configs")
    assert os.path.exists("models")
    assert os.path.exists("training")
    assert os.path.exists("evaluation")
