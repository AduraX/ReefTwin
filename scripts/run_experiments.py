"""CLI to run all experiments and generate report."""

from infrastructure.mlops.experiments import run_all_experiments, generate_report
from pathlib import Path


def main():
    results = run_all_experiments()
    report = generate_report(results)
    print(report)

    out = Path("data/gold/experiment_report.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
