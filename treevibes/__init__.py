"""Reproducible TreeVibes experiments with optional training backends."""
import os

# TFMOT uses Keras 2; choose it before any TensorFlow import in entry points.
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
__version__ = '0.1.0'
