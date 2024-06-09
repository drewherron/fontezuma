from fontTools.ttLib import TTFont
import re
import sys

# Get the name of a font from the TTF file
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

# For testing
if __name__ == "__main__":
    if len(sys.argv) > 1:
        font_name = get_sanitized_font_name(sys.argv[1])
        print(font_name)
    else:
        print("No argument provided.")
        sys.exit(1)
