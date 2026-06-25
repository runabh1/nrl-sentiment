"""
NRL Sentinel - AI Decision Intelligence System
Numaligarh Refinery Limited | Expansion Risk & Operational Analytics

Modules:
  - data_collector: Real data fetching + synthetic data generation
  - models: GRM, Demand, Maintenance, Cost Overrun, Pipeline Risk ML models
  - app: Streamlit interactive dashboard (5 modules)
"""

__version__ = "2.0.0"
__author__ = "NRL Operations Analytics Team"
__description__ = "AI-powered risk intelligence for NRL's 3→9 MMTPA expansion"

from .data_collector import (
    fetch_crude_and_product_prices,
    generate_ne_india_demand,
    generate_sensor_data,
    generate_project_data,
    generate_pipeline_data,
)

from .models import (
    train_grm_model,
    train_demand_model,
    train_maintenance_model,
    train_overrun_model,
    train_pipeline_model,
)

__all__ = [
    "fetch_crude_and_product_prices",
    "generate_ne_india_demand",
    "generate_sensor_data",
    "generate_project_data",
    "generate_pipeline_data",
    "train_grm_model",
    "train_demand_model",
    "train_maintenance_model",
    "train_overrun_model",
    "train_pipeline_model",
]
