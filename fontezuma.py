import argparse
from fz_image_processing import detect_and_normalize
from fz_predict_font import aggregate_predictions


# Allow command line arguments
def get_args():
    parser = argparse.ArgumentParser(
        prog="python3 fontezuma.py",
        description="Predict the font used in an image of text.")
    parser.add_argument("image_path",
        type=str,
        help="Path to the image file.")
    parser.add_argument("-v", "--verbose",
        action="store_true",
        help="Increase output verbosity.")
    return parser.parse_args()

def main(image_path, verbose=False):
    # Return list of characters from image processing
    if verbose:
        print("Starting text detection and normalization...")
    char_images = detect_and_normalize_text(image_path)

    if verbose:
        print(f"Detected {len(char_images)} characters.")

    # Predict on list of characters
    if verbose:
        print("Predicting fonts for each character...")
    aggregate_predictions(char_images)


if __name__ == "__main__":
    args = get_args()
    main(args.image_path, args.verbose)
