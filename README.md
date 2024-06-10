# fontezuma
A tool to detect text in an image, and name the font

1. Get your ttf files in a directory called 'fonts'
2. Run font2glyph.py to create images of every character
3. Run fz_cnn.py to train a neural network on those images
4. fontezuma.py takes an image path as argument, will make a font prediction
