import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Input, Embedding, Reshape
from sklearn.utils import shuffle
import os

# Load skin metadata
skin_df = pd.read_csv('HAM10000_metadata.csv')

# Handle missing values in 'age' by filling with the mean
skin_df['age'].fillna(skin_df['age'].mean(), inplace=True)

# Set up image file paths
image_dir = 'HAM10000_images_part_1'
images = []
for image_id in skin_df['image_id']:
    image_path = image_dir + image_id + '.jpg'
    image_path = os.path.join(image_dir, 'ISIC_0027419.jpg')
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image /= 255.0  # Normalize pixel values to [0, 1]
    images.append(image)

# Encode labels using LabelEncoder and shuffle data
label_encoder = LabelEncoder()
skin_df['encoded_label'] = label_encoder.fit_transform(skin_df['dx'])
images, labels = shuffle(images, skin_df['encoded_label'].values, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create a CNN model for image classification
model_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(7, activation='softmax')  # 7 classes in HAM10000 dataset
])

model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
history_cnn = model_cnn.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Create a sequence-to-sequence model with a pre-trained base model (VGG16)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

model_rnn = Sequential([
    base_model,
    Flatten(),
    Reshape((1, -1)),  # Reshape to a sequence
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(7, activation='softmax')  # Adjust the number of output classes as needed
])

model_rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create an image data generator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    rescale=1./255
)

# Train the sequence-to-sequence model with data augmentation
history_rnn = model_rnn.fit_generator(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=10,
    validation_data=(X_test, y_test)
)

# Evaluate the sequence-to-sequence model
test_loss, test_accuracy = model_rnn.evaluate(X_test, y_test)
print(f"Test Accuracy (RNN): {test_accuracy}")

# Plot training and validation accuracy for both models
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history_cnn.history['accuracy'], label='CNN Train')
plt.plot(history_cnn.history['val_accuracy'], label='CNN Validation')
plt.title('CNN Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_rnn.history['accuracy'], label='RNN Train')
plt.plot(history_rnn.history['val_accuracy'], label='RNN Validation')
plt.title('RNN Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()