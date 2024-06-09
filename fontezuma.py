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
    parser.add_argument("-e", "--export",
        action="store_true",
        help="Export processed character images for debugging.")
    parser.add_argument("-v", "--verbose",
        action="store_true",
        help="Increase output verbosity.")
    return parser.parse_args()

def main():
    args = get_args()
    image_path = args.image_path
    verbose = args.verbose
    export = args.export

    # Return list of characters from image processing
    if verbose:
        print("Starting text detection and normalization...")
    char_images = detect_and_normalize(image_path)

    if verbose:
        print(f"Detected {len(char_images)} characters.")

    if export:
        export_directory = 'export_chars'
        os.makedirs(export_directory, exist_ok=True)
        for i, img in enumerate(char_images):
            export_path = os.path.join(export_directory, f'char_{i}.png')
            cvt_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(export_path, cvt_image)
        if verbose:
            print(f"Exported character images to {export_directory}")

    # Predict on list of characters
    if verbose:
        print("Predicting fonts...")
    predictions = aggregate_predictions(char_images)

    # Print the top results
    print("Font predictions:")
    for font, score in sorted_fonts:
        print(f"{font}: {score:.4f}")


if __name__ == "__main__":
    main()
