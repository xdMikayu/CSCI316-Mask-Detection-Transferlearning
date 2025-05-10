# -*- coding: utf-8 -*-
"""
CSCI316 - Consolidated Face Mask Detection Project

This script combines functionality from three separate files:
1. Data download and preparation (untitled4.py)
2. Image augmentation techniques (untitled22.py)
3. Model training and evaluation (csci316_assignment_2_phase_3.py)
"""

# ===============================================================================
# SECTION 1: IMPORT LIBRARIES
# ===============================================================================

import os
import json
import shutil
import random
import zipfile
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ===============================================================================
# SECTION 2: DATASET DOWNLOAD AND PREPARATION
# ===============================================================================

# Configure Kaggle API
# kaggle_path = "/root/.kaggle"
# os.makedirs(kaggle_path, exist_ok=True)

# The following lines need to be run in a terminal or as shell commands
# !mv /content/kaggle.json {kaggle_path}/kaggle.json
# !chmod 600 {kaggle_path}/kaggle.json
# !pip install kaggle
# !kaggle datasets download -d andrewmvd/face-mask-detection --unzip -p /content/face-mask-dataset

# Define dataset paths
annotation_dir = "./face-mask-dataset/annotations"
image_dir = "./face-mask-dataset/images"
train_dir = "./face-mask-dataset/train"
split_dir = "./face-mask-dataset/split_data"

# Verify extracted files
print("Extracted files:", os.listdir("./face-mask-dataset"))

if os.path.exists(annotation_dir):
    files = os.listdir(annotation_dir)
    print("Annotation files:", files[:10])  # Show first 10 files
else:
    print("Annotations folder not found!")

# Explore XML structure
xml_file = "./face-mask-dataset/annotations/maksssksksss793.xml"
tree = ET.parse(xml_file)
root = tree.getroot()

print("XML structure:")
for child in root:
    print(child.tag, child.text)

# Extract labels from XML files
unique_labels = set()
for xml_file in os.listdir(annotation_dir):
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(annotation_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = root.findall("object")
        for obj in objects:
            label = obj.find("name").text.lower()
            unique_labels.add(label)

print("Unique labels in dataset:", unique_labels)

# Create class folders
categories = ["mask", "no_mask", "incorrect_mask"]
for category in categories:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)

# Mapping of XML labels to our class categories
label_map = {
    "with_mask": "mask",
    "without_mask": "no_mask",
    "mask_weared_incorrect": "incorrect_mask"
}

# Function to move images safely
def move_image_safe(src_path, dest_path):
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
        print(f"✅ MOVED: {src_path} → {dest_path}")
    else:
        print(f"❌ MISSING: {src_path}")

# Process all XML files and organize images into categories
for xml_file in os.listdir(annotation_dir):
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(annotation_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename = root.find("filename").text.strip()
        objects = root.findall("object")
        labels = [obj.find("name").text.lower() for obj in objects]
        
        print(f"📄 Processing {filename} - Labels: {labels}")
        
        # Convert XML labels to correct category
        assigned_label = None
        for label in labels:
            if label in label_map:
                assigned_label = label_map[label]
                break  # Stop once we find a valid label
                
        if assigned_label:
            src_path = os.path.join(image_dir, filename)
            dest_path = os.path.join(train_dir, assigned_label, filename)
            move_image_safe(src_path, dest_path)
        else:
            print(f"⚠️ Skipping {filename} - No valid label found")

print("\n✅ Image categorization complete!")

# Print count of images in each category
for category in categories:
    category_path = os.path.join(train_dir, category)
    print(f"📂 {category}: {len(os.listdir(category_path))} images")

# ===============================================================================
# SECTION 3: DATASET SPLITTING
# ===============================================================================

# Create train, val, test directories
split_types = ["train", "val", "test"]
for split_type in split_types:
    for category in categories:
        os.makedirs(os.path.join(split_dir, split_type, category), exist_ok=True)

# Define split ratios
split_ratio = {"train": 0.7, "val": 0.2, "test": 0.1}

# Split images for each category
for category in categories:
    category_path = os.path.join(train_dir, category)
    images = os.listdir(category_path)
    random.shuffle(images)
    
    # Compute split sizes
    total = len(images)
    train_count = int(total * split_ratio["train"])
    val_count = int(total * split_ratio["val"])
    
    # Assign images to each split
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]
    
    # Move images into respective directories
    for img_set, dest_type in zip([train_images, val_images, test_images], split_types):
        dest_path = os.path.join(split_dir, dest_type, category)
        for img in img_set:
            shutil.copy(os.path.join(category_path, img), os.path.join(dest_path, img))
    
    print(f"✅ {category}: {train_count} train, {val_count} val, {len(test_images)} test")

print("\n✅ Dataset split complete!")

# ===============================================================================
# SECTION 4: DATASET VISUALIZATION
# ===============================================================================

# Function to plot sample images
def plot_sample_images(directory, category, num_samples=5):
    category_path = os.path.join(directory, category)
    images = os.listdir(category_path)
    sample_images = random.sample(images, min(num_samples, len(images)))
    
    plt.figure(figsize=(10, 5))
    for i, img_name in enumerate(sample_images):
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(category)
    
    plt.show()

# Show sample images from training set
for category in categories:
    print(f"📸 Sample images from {category} category:")
    plot_sample_images(os.path.join(split_dir, "train"), category)

# Function to plot class distribution
def plot_class_distribution(directory, title):
    class_counts = {category: len(os.listdir(os.path.join(directory, category))) for category in categories}
    plt.figure(figsize=(8, 5))
    plt.bar(class_counts.keys(), class_counts.values(), color=['blue', 'orange', 'red'])
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

# Plot distributions for each dataset split
print("📊 Class Distribution in Training Set:")
plot_class_distribution(os.path.join(split_dir, "train"), "Training Set Distribution")

print("📊 Class Distribution in Validation Set:")
plot_class_distribution(os.path.join(split_dir, "val"), "Validation Set Distribution")

print("📊 Class Distribution in Test Set:")
plot_class_distribution(os.path.join(split_dir, "test"), "Test Set Distribution")

# ===============================================================================
# SECTION 5: IMAGE AUGMENTATION
# ===============================================================================

# Define augmented image storage directory
augmented_dir = "./face-mask-dataset/augmented_images"
os.makedirs(augmented_dir, exist_ok=True)

# OpenCV-based augmentation function
def augment_image_opencv(image):
    """Applies OpenCV-based augmentation: Random Rotation"""
    angle = random.choice([0, 90, 180, 270])
    rotation_map = {
        0: image,
        90: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(image, cv2.ROTATE_180),
        270: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    }
    return rotation_map[angle]

# Define Keras ImageDataGenerator for additional augmentations
datagen = ImageDataGenerator(
    shear_range=0.2,         # Shear transformation
    zoom_range=0.2,          # Random zoom
    horizontal_flip=True,    # Flip images horizontally
    brightness_range=[0.8, 1.2]  # Adjust brightness
)

print("Applying augmentations to training images...")

# Apply augmentations to each category in training set
for category in categories:
    category_path = os.path.join(split_dir, "train", category)
    category_aug_dir = os.path.join(augmented_dir, category)
    os.makedirs(category_aug_dir, exist_ok=True)
    
    for filename in os.listdir(category_path):
        img_path = os.path.join(category_path, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize to (128,128)
            
            for i in range(3):  # Generate 3 variations per image
                # OpenCV augmentation
                aug_img = augment_image_opencv(img)
                
                # Convert to TensorFlow tensor for additional augmentations
                aug_img = tf.convert_to_tensor(aug_img / 255.0, dtype=tf.float32)
                aug_img = tf.image.random_flip_left_right(aug_img)  # Flip horizontally
                aug_img = tf.image.random_brightness(aug_img, max_delta=0.2)  # Adjust brightness
                
                # Convert back to numpy for Keras processing
                aug_img = (aug_img.numpy() * 255).astype(np.uint8)
                
                # Expand dimensions for Keras
                aug_img_exp = np.expand_dims(aug_img, axis=0)
                
                # Apply Keras augmentations
                for batch in datagen.flow(aug_img_exp, batch_size=1):
                    aug_img_final = batch[0].astype(np.uint8)
                    break
                
                # Save augmented image
                aug_filename = f"aug_{i}_{filename}"
                aug_img_path = os.path.join(category_aug_dir, aug_filename)
                cv2.imwrite(aug_img_path, aug_img_final)

print("✅ Image augmentation complete!")

# ===============================================================================
# SECTION 6: MODEL DEVELOPMENT
# ===============================================================================

# Set up data generators with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    os.path.join(split_dir, "train"), 
    target_size=(128, 128), 
    batch_size=32, 
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(split_dir, "val"), 
    target_size=(128, 128), 
    batch_size=32, 
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(split_dir, "test"), 
    target_size=(128, 128), 
    batch_size=32, 
    class_mode='categorical', 
    shuffle=False
)

# Load InceptionV3 as the base model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Fine-tuning: freeze early layers, train later layers
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

# Add custom layers for face mask classification
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(3, activation='softmax')(x)

# Define the final model
model = Model(inputs=base_model.input, outputs=outputs)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Train model with early stopping
history = model.fit(
    train_generator, 
    epochs=20, 
    validation_data=val_generator, 
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
)

# ===============================================================================
# SECTION 7: MODEL EVALUATION
# ===============================================================================

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")

# Generate predictions
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Print classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=categories))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot training history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Save the model
model.save("face_mask_detector.h5")
print("✅ Model saved as 'face_mask_detector.h5'!")
