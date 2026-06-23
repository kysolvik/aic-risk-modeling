
import argparse
from .eval import calc_stats, load_preprocess_inputs

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate fire model outputs")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions file (tif OR csv with 'pred column'",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Path to ground truth file (tif OR csv with 'label' column)",
    )
    parser.add_argument(
        "--grouped",
        action="store_true",
        required=False,
        help="If provided, groups pred performance by values in ground_turth",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        help="Threshold for converting predictions to binary.",
        default=0.5,
    )
    parser.add_argument(
        "--reliability-plot",
        type=str,
        required=False,
        default=None,
        help="If provided, path to write a reliability-diagram PNG (binary only).",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        required=False,
        default=15,
        help="Number of equal-width probability bins for ECE / reliability.",
    )
    return parser.parse_args()

def main():
    """Main function to run evaluation"""
    args = parse_args()
    predictions, ground_truth = load_preprocess_inputs(args.predictions, args.ground_truth)
    # Calculate stats
    calc_stats(predictions, ground_truth, grouped=args.grouped, threshold=args.threshold,
               reliability_plot=args.reliability_plot,
               calibration_bins=args.calibration_bins)

if __name__ == "__main__":
    main()
