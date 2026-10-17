# Bundled synthetic data

`subscriptions.csv` is generated locally by `data/generate_sample.py` using Python's seeded pseudo-random generator. It contains 800 fictional subscribers. No real customer records or third-party dataset are included.

| Column | Type | Meaning |
| --- | --- | --- |
| `tenure_months` | Numeric | Synthetic subscription duration, 1–60 months |
| `monthly_charge` | Numeric | Generated from the selected plan's fixed sample price |
| `support_tickets` | Numeric | Sampled count of support contacts |
| `weekly_usage_hours` | Numeric | Simulated weekly use with plan-dependent average |
| `plan` | Categorical | starter, standard, or pro |
| `region` | Categorical | One of four fictional regions |
| `autopay` | Categorical | Whether simulated automatic payment is active |
| `churned` | Binary target | Generated from a noisy probability influenced by tenure, plan, support contacts, usage, and auto-pay |

The features and target are intentionally correlated so a first benchmark is interesting. Since the data is synthetic, model scores only demonstrate the tool's behavior; they say nothing about real customer churn.
