# model_training.py

# -------------------------
# 1. Import dependencies
# -------------------------
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# -------------------------
# 2. Define dataset paths
# -------------------------
train_dir = os.path.join('dataset', 'train')
test_dir = os.path.join('dataset', 'test')

# -------------------------
# 3. Image preprocessing
# -------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# -------------------------
# 4. Build CNN model
# -------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral
])

# -------------------------
# 5. Compile the model
# -------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------
# 6. Train the model
# -------------------------
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator
)

# -------------------------
# 7. Save the model
# -------------------------
model.save('face_emotionModel.h5')
print("✅ Model training complete and saved as face_emotionModel.h5")
