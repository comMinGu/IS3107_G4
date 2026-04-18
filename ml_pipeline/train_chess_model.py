"""
Placeholder entry point for the monthly chess model retraining pipeline.

This script is intentionally minimal for now. The notebook logic in
`ml_model.ipynb` will later be refactored into functions that can be called
from here by Airflow.
"""


def main(run_date: str | None = None) -> None:
    print("Monthly chess model retraining entry point")
    print(f"run_date={run_date}")
    print("TODO: move notebook training logic here for Airflow execution.")


if __name__ == "__main__":
    main()
