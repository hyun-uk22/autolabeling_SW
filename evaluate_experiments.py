import argparse
import os

from src.utils.evaluation import build_experiment_report, save_experiment_report


def parse_runs(values):
    runs = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Run must be formatted as name=directory: {value}")
        name, path = value.split("=", 1)
        runs[name] = path
    return runs


def main():
    parser = argparse.ArgumentParser(description="Build quantitative ablation reports from run output directories.")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories as name=path, e.g. low=data/runs/low cascade=data/runs/cascade",
    )
    parser.add_argument("--gt_dir", default=None, help="Optional YOLO ground-truth directory")
    parser.add_argument("--out_dir", default="data/reports", help="Output directory for report files")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for detection evaluation")
    parser.add_argument("--manual_time_per_image", type=float, default=45.0, help="Manual labeling time baseline in seconds")
    parser.add_argument("--low_unit_cost", type=float, default=1.0, help="Relative unit cost for low model API calls")
    parser.add_argument("--high_unit_cost", type=float, default=10.0, help="Relative unit cost for high model API calls")
    args = parser.parse_args()

    runs = parse_runs(args.runs)
    report = build_experiment_report(
        runs,
        gt_dir=args.gt_dir,
        iou_threshold=args.iou,
        manual_time_per_image=args.manual_time_per_image,
        low_unit_cost=args.low_unit_cost,
        high_unit_cost=args.high_unit_cost,
    )
    paths = save_experiment_report(report, args.out_dir)

    print("Experiment report saved:")
    for name, path in paths.items():
        print(f" - {name}: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
