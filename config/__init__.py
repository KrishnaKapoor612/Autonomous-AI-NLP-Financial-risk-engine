"""
Config Package Initialization.

Purpose:
  Exports all configuration constants and tunable parameters from config.settings
  to provide centralized access for the rest of the application.

Role in Project:
  Enables any module across ingestion, NLP engine, quantitative modules, and
  dashboards to cleanly import shared thresholds, portfolio baselines, and model paths.
"""

from config.settings import *
