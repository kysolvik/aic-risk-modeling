
import argparse
from .eval import calc_stats, load_preprocess_inputs, write_calibrated_predictions

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
        help="Number of probability bins for ECE / reliability.",
    )
    parser.add_argument(
        "--calibration-binning",
        choices=("uniform", "quantile"),
        default="uniform",
        help="ECE bin spacing: 'uniform' (equal-width) or 'quantile' "
             "(equal-count). Quantile keeps the sparse high-probability region "
             "from being swamped by near-zero pixels under heavy imbalance.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=("none", "temperature", "platt", "isotonic"),
        default="none",
        help="Post-hoc calibrator to fit and apply (binary only). 'temperature' "
             "= single scalar; 'platt' = logistic with intercept (fixes bias); "
             "'isotonic' = non-parametric monotonic map.",
    )
    parser.add_argument(
        "--calibration-fit-predictions",
        type=str,
        default=None,
        help="Predictions of a held-out calibration set to FIT the calibrator on "
             "(applied to --predictions). Out-of-sample alternative to in-sample fitting.",
    )
    parser.add_argument(
        "--calibration-fit-ground-truth",
        type=str,
        default=None,
        help="Ground truth for --calibration-fit-predictions.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=None,
        help="Apply temperature scaling with this known T directly, without "
             "fitting (binary only). Overrides --calibration-method.",
    )
    parser.add_argument(
        "--calibrated-output",
        type=str,
        required=False,
        default=None,
        help="If set, write the calibrated predictions here, mirroring the input "
             "format (tif->tif preserving georeferencing, csv->csv 'pred' column). "
             "Requires a calibration method or --temperature (binary only).",
    )
    return parser.parse_args()

def main():
    """Main function to run evaluation"""
    args = parse_args()
    predictions, ground_truth = load_preprocess_inputs(args.predictions, args.ground_truth)

    if bool(args.calibration_fit_predictions) != bool(args.calibration_fit_ground_truth):
        raise SystemExit("--calibration-fit-predictions and "
                         "--calibration-fit-ground-truth must be given together.")
    calibration_fit = None
    if args.calibration_fit_predictions:
        calibration_fit = load_preprocess_inputs(
            args.calibration_fit_predictions, args.calibration_fit_ground_truth)

    method = args.calibration_method

    # Calculate stats
    _, calibrated = calc_stats(
        predictions, ground_truth, grouped=args.grouped, threshold=args.threshold,
        reliability_plot=args.reliability_plot,
        calibration_bins=args.calibration_bins,
        calibration_binning=args.calibration_binning,
        calibration_method=method, calibration_fit=calibration_fit,
        temperature=args.temperature)

    if args.calibrated_output:
        if calibrated is None:
            raise SystemExit(
                "--calibrated-output requires a calibration method or --temperature "
                "to be applied (binary predictions only).")
        write_calibrated_predictions(args.predictions, args.calibrated_output, calibrated)

if __name__ == "__main__":
    main()
