import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


# ✅ Path to your dataset
data_dir = r'C:\Users\user\Desktop\Production Project\LifePulse\Clean_Dataset\Malaria_cell_images\cell_images'

# ✅ Image parameters
img_width, img_height = 64, 64
batch_size = 32
epochs = 20  # Increase for better accuracy

# ✅ Data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# ✅ CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# ✅ Compile
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Train
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# ✅ Save model
save_path = r'C:\Users\user\Desktop\Production Project\LifePulse\models\malaria.h5'
model.save(save_path)
print(f"✅ Model saved to: {save_path}")
