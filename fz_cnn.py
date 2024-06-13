import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os


# Count the number of classes based on folders in the specified directory
def count_classes(directory):
    return len([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])

# Set up data generators for training and validation with data augmentation parameters
def create_generators(directory, batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        directory,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        directory,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

# Define and compile the CNN model with convolutional, max pooling, flatten, and dense layers
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Use the adam optimizer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    glyph_dir = './glyphs'
    num_classes = count_classes(glyph_dir)
    if num_classes == 0:
        raise ValueError("No character image files detected.")

    # Initialize training and validation generators and create the model
    train_generator, validation_generator = create_generators(glyph_dir, batch_size=32)
    model = create_model((64, 64, 3), num_classes)
    model.summary()

    # Train the model
    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size
        )
    except Exception as e:
        print(f"An error occurred: {e}")

    # Save the model
    model.save('font_recognition_model.keras')
    with open('class_indices.json', 'w') as json_file:
        json.dump(train_generator.class_indices, json_file)

if __name__ == "__main__":
    main()
