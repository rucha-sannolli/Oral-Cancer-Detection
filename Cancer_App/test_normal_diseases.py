import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("oral_cancer_model.h5")
print("Model loaded successfully.")

# Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
normal_diseases_test_path = "D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/Cancer_App/dataset/normal_diseases_test"

# Data preprocessing (rescaling)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load the normal oral diseases test set
normal_diseases_generator = val_test_datagen.flow_from_directory(
    normal_diseases_test_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,  # No labels, as we are testing
    shuffle=False
)

# Predict on the normal oral diseases test set
predictions = (model.predict(normal_diseases_generator) > 0.5).astype("int32").flatten()

# Analyze predictions and display results
for i, prediction in enumerate(predictions):
    label = "Cancer" if prediction == 1 else "Non-Cancer"
    print(f"Image {i+1}: {label}")
