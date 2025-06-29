import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
DATASET_PATH = "D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/Cancer_App/dataset"

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load Datasets
train_generator = train_datagen.flow_from_directory(
    f"{DATASET_PATH}/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_test_datagen.flow_from_directory(
    f"{DATASET_PATH}/val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    f"{DATASET_PATH}/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Compute Class Weights (in case of imbalance)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu'),  # New extra layer
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # For binary classification
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop, lr_reduce]
)

# Save Model
model.save("oral_cancer_model.h5")
print(" Model saved as oral_cancer_model.h5")

# Evaluate on Test Set
y_true = test_generator.classes
y_pred = (model.predict(test_generator) > 0.5).astype("int32").flatten()

# Metrics
conf_matrix = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=["Non-Cancer", "Cancer"])

print(f"\n Test Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Non-Cancer", "Cancer"],
            yticklabels=["Non-Cancer", "Cancer"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()










# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# # Define constants
# IMG_SIZE = (224, 224)  # Resize all images to 224x224
# BATCH_SIZE = 32
# EPOCHS = 10
# DATASET_PATH = "D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/Cancer_App/dataset"  # Change this to your dataset location

# # Data augmentation and normalization for training set
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# # Only normalization for validation and test datasets
# val_test_datagen = ImageDataGenerator(rescale=1./255)

# # Load datasets
# train_generator = train_datagen.flow_from_directory(
#     f"{DATASET_PATH}/train",
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='binary'
# )

# val_generator = val_test_datagen.flow_from_directory(
#     f"{DATASET_PATH}/val",
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='binary'
# )

# test_generator = val_test_datagen.flow_from_directory(
#     f"{DATASET_PATH}/test",
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     shuffle=False
# )

# # Build CNN model
# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
#     MaxPooling2D(2,2),

#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D(2,2),

#     Conv2D(128, (3,3), activation='relu'),
#     MaxPooling2D(2,2),

#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')  # Binary classification
# ])

# # Compile model
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # Train model
# history = model.fit(train_generator,
#                     validation_data=val_generator,
#                     epochs=EPOCHS)

# # Save the trained model
# model.save("oral_cancer_model.h5")
# print("Model saved as oral_cancer_model.h5")

# # Evaluate model
# y_true = test_generator.classes  # Actual labels
# y_pred = (model.predict(test_generator) > 0.5).astype("int32").flatten()

# # Confusion Matrix & Metrics
# conf_matrix = confusion_matrix(y_true, y_pred)
# accuracy = accuracy_score(y_true, y_pred)
# report = classification_report(y_true, y_pred, target_names=["Non-Cancer", "Cancer"])

# print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")
# print("Classification Report:\n", report)

# # Plot confusion matrix
# plt.figure(figsize=(6,5))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#             xticklabels=["Non-Cancer", "Cancer"],
#             yticklabels=["Non-Cancer", "Cancer"])
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()



# import os
# from PIL import Image

# def remove_corrupted_images(directory):
#     """Check and remove corrupted images in a given directory."""
#     for folder in os.listdir(directory):
#         folder_path = os.path.join(directory, folder)
#         if not os.path.isdir(folder_path):
#             continue
#         for filename in os.listdir(folder_path):
#             file_path = os.path.join(folder_path, filename)
#             try:
#                 img = Image.open(file_path)  # Try to open image
#                 img.verify()  # Verify image integrity
#             except (IOError, SyntaxError):
#                 print(f"Removing corrupted image: {file_path}")
#                 os.remove(file_path)  # Remove corrupted file

# # Apply to train, val, and test folders
# dataset_path = "D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/Cancer_App/dataset"  # Change this to your dataset path
# remove_corrupted_images(os.path.join(dataset_path, "train"))
# remove_corrupted_images(os.path.join(dataset_path, "val"))
# remove_corrupted_images(os.path.join(dataset_path, "test"))

# print("Corrupted images removed successfully!")
