import os
import tensorflow as tf #type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore

train_dir = '../data/train'
val_dir = '../data/validation'

train_gen = ImageDataGenerator(rescale=1./255, rotation_range= 20, zoom_range = 0.2, horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=(150, 150), batch_size= 32, class_mode="binary")
val_data = val_gen.flow_from_directory(val_dir, target_size=(150, 150), batch_size= 32, class_mode="binary")

model = Sequential([
    Conv2D(32, (3,3), activation= 'relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation= 'relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation= 'relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer = Adam(), loss= 'binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, validation_data=val_data, epochs=10)

model.save('models/cat_dog_classifier.h5')

print("Model trained and saved successfully.")
