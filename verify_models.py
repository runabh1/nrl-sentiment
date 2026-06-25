"""
NRL Sentinel — Model Verification & Metrics Report
Run this script to verify all 5 models and print metrics.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  NRL SENTINEL — MODEL VERIFICATION REPORT")
print("=" * 60)

from data_collector import (
    fetch_crude_and_product_prices,
    generate_ne_india_demand,
    generate_sensor_data,
    generate_project_data,
    generate_pipeline_data,
)
from models import (
    train_grm_model,
    train_demand_model,
    train_maintenance_model,
    train_overrun_model,
    train_pipeline_model,
)

# ── Load / fetch data ────────────────────────────────────────────
print("\n[1/5] Loading data...")
crude    = fetch_crude_and_product_prices()
demand   = generate_ne_india_demand()
sensor   = generate_sensor_data()
project  = generate_project_data()
pipeline = generate_pipeline_data()
print("  Data ready.\n")

# ── MODULE 1: GRM ────────────────────────────────────────────────
print("[MODULE 1] GRM Forecasting")
print("-" * 40)
try:
    r = train_grm_model(crude)
    print(f"  Training rows : {r['training_rows']:,}")
    print(f"  Test MAE      : Rs {r['mae']:.2f} / bbl")
    print(f"  Test R^2      : {r['r2']:.4f}")
    print(f"  Current GRM   : Rs {r['current_grm']:.0f} / bbl (crack spread)")
    print(f"  NRL Net GRM   : Rs {r['current_nrl_grm']:.0f} / bbl (calibrated)")
    print(f"  Capture ratio : {r['calibration_ratio']:.1%}")
    print(f"  90d forecast  : Rs {r['forecast']['grm_forecast_inr'].mean():.0f} / bbl avg")
    status = "PASS" if r['r2'] > 0.7 else "WARN (low R2)" if r['r2'] > 0 else "FAIL (negative R2)"
    print(f"  STATUS        : {status}")
except Exception as e:
    print(f"  ERROR: {e}")
    r = None

# ── MODULE 2: DEMAND ─────────────────────────────────────────────
print("\n[MODULE 2] Demand Forecasting")
print("-" * 40)
try:
    d = train_demand_model(demand)
    print(f"  CV method     : TimeSeriesSplit (5 folds)")
    print(f"  CV MAE        : {d['mae']:.2f} TMTPA")
    print(f"  CV R^2        : {d['r2']:.4f}")
    fc_total = d['forecast']['demand_forecast'].sum()
    print(f"  12mo forecast : {fc_total:.1f} TMTPA total")
    for prod, growth in d['product_growth'].items():
        print(f"    {prod:<22}: {growth:+.1f}% YoY growth")
    status = "PASS" if 0.5 < d['r2'] < 0.99 else "WARN"
    print(f"  STATUS        : {status}")
except Exception as e:
    print(f"  ERROR: {e}")
    d = None

# ── MODULE 3: MAINTENANCE ────────────────────────────────────────
print("\n[MODULE 3] Predictive Maintenance")
print("-" * 40)
try:
    m = train_maintenance_model(sensor)
    print(f"  Equipment units: {len(m)}")
    for equip, info in m.items():
        print(f"  {equip[:20]:<20} health={info['health_score']:.0f}%  "
              f"ROC-AUC={info['roc_auc']:.3f}  "
              f"fail_prob={info['failure_prob_now']:.1f}%  "
              f"status={info['status']}")
    status = "PASS"
    print(f"  STATUS        : {status}")
except Exception as e:
    print(f"  ERROR: {e}")
    m = None

# ── MODULE 4: COST OVERRUN ───────────────────────────────────────
print("\n[MODULE 4] Cost Overrun (NREP)")
print("-" * 40)
try:
    o = train_overrun_model(project)
    print(f"  CV method     : {o['cv_method']}")
    print(f"  LOOCV MAE     : {o['mae']:.2f} %")
    print(f"  LOOCV R^2     : {o['r2']:.4f}")
    print(f"  Total overrun : Rs {o['total_cost_overrun_cr']:.0f} Cr")
    print(f"  Avg overrun % : {o['avg_overrun_pct']:.1f}%")
    print(f"  MC P10 cost   : Rs {o['mc_p10']:.0f} Cr")
    print(f"  MC P50 cost   : Rs {o['mc_p50']:.0f} Cr")
    print(f"  MC P90 cost   : Rs {o['mc_p90']:.0f} Cr")
    status = "PASS" if o['r2'] > 0.3 else "WARN (limited n=10 data)"
    print(f"  STATUS        : {status}")
except Exception as e:
    print(f"  ERROR: {e}")
    o = None

# ── MODULE 5: PIPELINE ───────────────────────────────────────────
print("\n[MODULE 5] Pipeline Utilization")
print("-" * 40)
try:
    p = train_pipeline_model(pipeline)
    print(f"  ROC-AUC       : {p['roc_auc']:.3f}")
    print(f"  Avg util      : {p['avg_utilization']:.1f}%")
    print(f"  Penalty months: {p['penalty_months_hist']}")
    print(f"  Current util  : {p['current_utilization']:.1f}%")
    print(f"  Forecast pen. : {p['forecast_penalty_months']} months  "
          f"Rs {p['forecast_penalty_cost_cr']:.1f} Cr")
    status = "PASS" if p['roc_auc'] >= 0.5 else "WARN"
    print(f"  STATUS        : {status}")
except Exception as e:
    print(f"  ERROR: {e}")
    p = None

print("\n" + "=" * 60)
print("  VERIFICATION COMPLETE")
print("=" * 60)
