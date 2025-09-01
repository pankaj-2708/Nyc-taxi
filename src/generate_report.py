import mlflow
import pandas as pd

# Search for runs within a specific experiment
runs = mlflow.search_runs(experiment_names=["Default"])

# Sort by the r2_score metric in descending order (best first)
runs.sort_values(by="metrics.r2_score", ascending=False, inplace=True)

# Get the best run
best_run = runs.iloc[0]

# Write a formatted Markdown report
report_path = './report.md'
with open(report_path, 'w') as f:
    f.write(f"# MLflow Experiment Report\n\n")
    f.write(f"## Best Run Metrics and Parameters\n\n")
    
    # Write metrics and parameters
    for col in runs.columns:
        f.write(f"- **{col}**: {best_run[col]}\n")