
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
    return parser.parse_args()

def main():
    """Main function to run evaluation"""
    args = parse_args()
    predictions, ground_truth = load_preprocess_inputs(args.predictions, args.ground_truth)
    # Calculate stats
    calc_stats(predictions, ground_truth, grouped=args.grouped, threshold=args.threshold)

if __name__ == "__main__":
    main()
