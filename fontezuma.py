import argparse
import cv2
import os
import numpy as np
from fz_image_processing import detect_and_normalize



# Allow command line arguments
def get_args():
    parser = argparse.ArgumentParser(
        prog="python3 fontezuma.py",
        description="Predict the font used in an image of text.")
    parser.add_argument("image_path",
        type=str,
        help="Path to the image file.")
    parser.add_argument("-e", "--export",
        action="store_true",
        help="Export processed character images for debugging.")
    parser.add_argument("-s", "--score",
        action="store_true",
        help="Display scores along with font predictions.")
    parser.add_argument("-n",
        type=int,
        default=1,
        help="Number of predictions to display.")
    parser.add_argument("-v", "--verbose",
        action="store_true",
        help="Increase output verbosity.")
    return parser.parse_args()

def main():
    args = get_args()
    image_path = args.image_path
    verbose = args.verbose
    export = args.export
    show_scores = args.score
    num_predictions = args.n

    # Control TensorFlow verbosity
    if not verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    # Importing here to control verbosity
    from fz_predict_font import predict_font, aggregate_predictions

    # Get list of characters from image processing
    if verbose:
        print(f"\nInput image: {image_path}")
        print("Starting text detection and normalization...")
    try:
        char_images = detect_and_normalize(image_path, export, verbose)
        if verbose:
            if char_images:
                print(f"\nDetected {len(char_images)} characters.")
            else:
                print("No characters were detected.")

    except Exception as e:
        print(f"Error during text detection and normalization: {e}")
        return

    # Predict on list of characters
    if verbose:
        print("Predicting fonts...")
    try:
        predictions = [predict_font(img) for img in char_images if isinstance(img, np.ndarray)]
        agg_predictions = aggregate_predictions(predictions)
    except Exception as e:
        print(f"Error during font prediction: {e}")
        return

    if num_predictions == 1 and agg_predictions:
        if show_scores:
            print(f"\nFont prediction: {agg_predictions[0][0]}\nscore: {agg_predictions[0][1]}\n")
        else:
            print(agg_predictions[0][0])

    elif num_predictions > 1 and agg_predictions:
        print("\nFont predictions:")
        for font, score in agg_predictions[:num_predictions]:
            if show_scores:
                print(f"{font}: {score:.4f}")
            else:
                print(f"{font}")
        print()
    else:
        print("\nNo predictions were made.\n")

if __name__ == "__main__":
    main()
