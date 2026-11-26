# -*- coding: utf-8 -*-
"""LSTM-based movie review sentiment analysis system."""
from .config import Config
from .data import clean_text, prepare_data, tokenize
from .model import build_model, load_model

__all__ = ["Config", "clean_text", "tokenize", "prepare_data", "build_model", "load_model"]
__version__ = "1.0.0"
