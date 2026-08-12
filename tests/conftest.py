"""Puts the project root on sys.path so the tests can import the engine modules
(coffeetech_rules_v5, coffeetech_doses, coffeetech_recommendations) without installing the package
or the heavy dependencies (FastAPI, pandas, sklearn)."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
