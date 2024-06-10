import os
from PIL import ImageFont, ImageDraw, Image
from fontTools.ttLib import TTFont
import re
import sys

# Get the name of a font from the TTF metadata
def get_sanitized_font_name(font_path):
    font = TTFont(font_path)
    for record in font['name'].names:
        # NameID 4 is the full font name
        if record.nameID == 4:
            font_name = record.toUnicode()
            # Remove illegal characters
            sanitized_name = re.sub(r'[<>:"/\\|?*]', '', font_name)
            # Replace spaces with dashes
            sanitized_name = sanitized_name.replace(' ', '-')
            return sanitized_name
    return None

# Supply a font name and a char, get an image
def render_glyph_from_font(font_path, char, output_dir, font_size=300):
    font_name = get_sanitized_font_name(font_path)
    if font_name is None:
        print(f"Font name not found for {font_path}")
        return

    font = ImageFont.truetype(font_path, font_size)
    # Create a transparent image
    image = Image.new('RGBA', (400, 400), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    # Draw the character in black
    draw.text((50, 50), char, font=font, fill=(0, 0, 0, 255))

    # Crop the image to the char
    bbox = image.getbbox()
    print(f"bbox: {bbox}")
    if bbox:
        image = image.crop(bbox)

    # Make grayscale
    image = image.convert('L')

    # Make square
    image = image.resize((200, 200), Image.LANCZOS)

    # Ensure the output directory exists
    char_dir = os.path.join(output_dir, font_name)
    if not os.path.exists(char_dir):
        os.makedirs(char_dir)

    # Save the image
    image.save(os.path.join(char_dir, f'{char}.png'))
    print(f"Saved glyph at: {char_dir}/{char}.png")

# Makes a glyph for every char, for every font
def process_all_fonts(root_dir):
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.ttf'):
                font_path = os.path.join(subdir, file)
                output_dir = './glyphs'
                for char in chars:
                    print(f"Processing char: {char} from font: {font_path}")
                    render_glyph_from_font(font_path, char, output_dir)


if __name__ == "__main__":
    # Optional font directory as argument
    if len(sys.argv) > 1:
        font_directory = sys.argv[1]
    # Otherwise use default
    else:
        font_directory = './fonts'

    process_all_fonts(font_directory)
