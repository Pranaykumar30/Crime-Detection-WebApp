import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def load_data(image_dir, label_dir):
    images, labels = [], []
    
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Skipping {img_file} - invalid image")
            continue
        
        label_file = img_file.replace(".jpg", ".txt")
        label_path = os.path.join(label_dir, label_file)
        
        if not os.path.exists(label_path):
            print(f"Skipping {img_file} - no label file")
            continue
        
        with open(label_path, "r") as f:
            lines = f.readlines()
            if not lines:
                print(f"Skipping {img_file} - empty label file")
                continue
            
            label = int(lines[0].split()[0])
        
        img = cv2.resize(img, (224, 224))
        images.append(img)
        labels.append(label)
    
    return np.array(images), tf.keras.utils.to_categorical(labels, num_classes=6)

def train_mobilenet():
    train_images, train_labels = load_data("/workspaces/Crime-Detection-WebApp/data/split/train/images", "/workspaces/Crime-Detection-WebApp/data/split/train/labels")
    val_images, val_labels = load_data("/workspaces/Crime-Detection-WebApp/data/split/val/images", "/workspaces/Crime-Detection-WebApp/data/split/val/labels")

    if len(train_images) == 0 or len(val_images) == 0:
        raise ValueError("No valid training or validation data found.")

    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(6, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=20, batch_size=8)

    os.makedirs("models/mobilenet", exist_ok=True)
    model.save("models/mobilenet/mobilenet_crime.h5")

if __name__ == "__main__":
    train_mobilenet()
