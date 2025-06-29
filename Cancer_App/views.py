from django.shortcuts import render, redirect
from django.http import HttpResponse, request
from django.contrib.auth.forms import AuthenticationForm
from .forms import RegistrationForm
from django.contrib.auth import login, logout, authenticate

from django.contrib.auth.decorators import login_required
from django.contrib import messages
import csv
from django.conf import settings
import os
from django.contrib.auth.decorators import login_required,user_passes_test

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from django.shortcuts import render
from io import BytesIO  # Import BytesIO
import traceback  # Import traceback for debugging

from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO

# Load CNN model once globally
cnn_model_path = "D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/oral_cancer_model.h5"
cnn_model = load_model(cnn_model_path)

def predict(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            img_file = request.FILES['image']
            img_bytes = BytesIO(img_file.read())
            img = image.load_img(img_bytes, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

            # Make sure to use cnn_model, not model
            prediction = cnn_model.predict(img_array)[0][0]
            result = "Positive (Cancer Detected)" if prediction < 0.5 else "Negative (No Cancer Detected)"
            return render(request, 'result.html', {'ans': result})

        except Exception as e:
            return render(request, 'result.html', {'ans': f"Error processing image: {str(e)}"})

    return render(request, 'predict.html')






# Load the trained SVM model and scaler (you would save your trained model earlier)
# svm_model = joblib.load('D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/svm_model_multi_class.pkl')
# scaler = joblib.load('D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/scaler.pkl')  # Load the fitted StandardScaler


import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from django.shortcuts import render

def predict_stage(request):
    if request.method == "POST":
        #  Step 1: Collect input data from form
        raw_data = {
            "Age": int(request.POST.get("age")),
            "Gender": request.POST.get("gender"),
            "Smoking History": request.POST.get("smoking_history"),  # Already 'Yes' or 'No'
            "Alcohol Consumption": request.POST.get("alcohol_consumption") or "None",  # Correct mapping
            "Tumor Size (cm)": float(request.POST.get("tumor_size")),
            "Lymph Node Involvement": request.POST.get("lymph_node_involvement"),
            "Weight Loss": request.POST.get("weight_loss") or "None",  # Already 'Moderate' or 'Severe'
            "Mouth Sores": request.POST.get("mouth_sores"),
        }

        #  Step 2: Create DataFrame
        input_df = pd.DataFrame([raw_data])

        #  Step 3: Handle missing values (if any)
        imputer = SimpleImputer(strategy='most_frequent')
        input_df = pd.DataFrame(imputer.fit_transform(input_df), columns=input_df.columns)

        #  Step 4: One-hot encoding (match training method)
        input_encoded = pd.get_dummies(input_df, drop_first=False)

        #  Step 5: Load saved model assets
        svm_model = joblib.load('D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/svm_model_multi_class_rbf.pkl')
        scaler = joblib.load('D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/scaler.pkl')
        feature_names = joblib.load('D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/feature_names.pkl')

        #  Step 6: Align input features with training features
        for col in feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[feature_names]  # Reorder to match training


        #  Step 7: Scale inputs
        input_scaled = scaler.transform(input_encoded)

        #  Step 8: Predict
        prediction = svm_model.predict(input_scaled)
        predicted_stage = prediction[0]

        # alcohol = request.POST.get("alcohol_consumption")
        # if alcohol is None or alcohol == "":
        #     alcohol = "None"  # default value or raise error

        # weight = request.POST.get("weight_loss")
        # if weight is None or weight == "":
        #     weight = "None"
        
        # for field in ["Smoking History", "Alcohol Consumption", "Weight Loss"]:
        #     if raw_data[field] is None or raw_data[field] == "":
        #         raw_data[field] = "None"

        print("\nRaw Data:\n", raw_data)
        print("\nEncoded Input:\n", input_encoded.head())
        print("\nPrediction:\n", prediction)

        #  Step 9: Return result
        return render(request, 'stage_result.html', {'predicted_stage': predicted_stage})

    return render(request, 'predict_stage.html')


###################### previous
# from sklearn.impute import SimpleImputer
# import pandas as pd
# import joblib
# from django.shortcuts import render

# # Load the trained model, scaler, and feature names
# svm_model = joblib.load('D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/svm_model_multi_class_rbf.pkl')
# scaler = joblib.load('D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/scaler.pkl')
# feature_names = joblib.load('D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/feature_names.pkl')

# def predict_stage(request):
#     if request.method == "POST":
#         # Step 1: Get raw inputs from POST request
#         raw_data = {
#             "Age": int(request.POST.get("age")),
#             "Gender": request.POST.get("gender"),
#             "Smoking History": "Yes" if request.POST.get("smoking_history") == "Smoker" else "No",
#             "Alcohol Consumption": "Moderate" if request.POST.get("alcohol_consumption") == "Yes" else "None",
#             "Tumor Size (cm)": float(request.POST.get("tumor_size")),
#             "Lymph Node Involvement": request.POST.get("lymph_node_involvement"),
#             "Weight Loss": "Severe" if request.POST.get("weight_loss") == "Yes" else "None",
#             "Mouth Sores": request.POST.get("mouth_sores"),
#         }

#         # Step 2: Create DataFrame from raw data
#         input_df = pd.DataFrame([raw_data])

#         # Step 3: Handle missing values (if any)
#         imputer = SimpleImputer(strategy='most_frequent')
#         input_df = pd.DataFrame(imputer.fit_transform(input_df), columns=input_df.columns)

#         # Step 4: One-hot encode features like during training
#         input_encoded = pd.get_dummies(input_df)

#         # Step 5: Ensure input matches the features used during training
#         # Ensure all training features exist in the input data
#         for col in feature_names:
#             if col not in input_encoded.columns:
#                 input_encoded[col] = 0
#         input_encoded = input_encoded[feature_names]

#         # Debugging: Check input_encoded
#         print("\nInput Encoded Features:")
#         print(input_encoded)

#         # Step 6: Scale the features
#         input_scaled = scaler.transform(input_encoded)

#         # Step 7: Predict the stage using the trained model
#         prediction = svm_model.predict(input_scaled)

#         # Debugging: Check prediction result
#         print("\nPredicted Stage:")
#         print(prediction)

#         # Step 8: Return the result to the user
#         predicted_stage = prediction[0]
#         return render(request, 'stage_result.html', {'predicted_stage': predicted_stage})

#     return render(request, 'predict_stage.html')
###################### previous 





#####
# #Load the pre-trained SVM model, scaler, and feature names
# model = joblib.load('D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/svm_model_multi_class.pkl')
# scaler = joblib.load('D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/scaler.pkl')
# feature_names = joblib.load('D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/feature_names.pkl')  # Load saved feature names

# def predict_stage(request):
#     if request.method == "POST":
#         # Extract form data from POST request
#         input_data = {
#             f"Age_{request.POST.get('age', 0)}": 1,
#             f"Tumor Size (cm)_{float(request.POST.get('tumor_size', 0.0))}": 1,
#             "Gender_Male": 1 if request.POST.get("gender") == "Male" else 0,
#             "Smoking History_Yes": 1 if request.POST.get("smoking_history") == "Smoker" else 0,
#             "Alcohol Consumption_Moderate": 1 if request.POST.get("alcohol_consumption") == "Yes" else 0,
#             "Lymph Node Involvement_Yes": 1 if request.POST.get("lymph_node_involvement") == "Yes" else 0,
#             "Weight Loss_Severe": 1 if request.POST.get("weight_loss") == "Yes" else 0,
#             "Mouth Sores_Yes": 1 if request.POST.get("mouth_sores") == "Yes" else 0,
#         }

#         # Load scaler, model, and feature names
#         scaler = joblib.load('D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/scaler.pkl')
#         svm_model = joblib.load('D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/svm_model_multi_class.pkl')
#         feature_names = joblib.load('D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/feature_names.pkl')

#         # Initialize input DataFrame with all feature columns set to 0, then update activated features
#         input_data_full = {feature: 0 for feature in feature_names}
#         input_data_full.update(input_data)

#         # Create a DataFrame from the input dictionary, aligned with model feature names
#         input_df = pd.DataFrame([input_data_full])

#         # Scale input and make a prediction
#         X_scaled = scaler.transform(input_df)
#         prediction = svm_model.predict(X_scaled)

#         # Get predicted stage and return the result page
#         predicted_stage = f"Stage {prediction[0]}"
#         return render(request, 'stage_result.html', {'predicted_stage': predicted_stage})

#     # If GET request, return the input form page
#     return render(request, 'predict_stage.html')
###########

# def result(request):
#     prediction = request.session.get('prediction', "No prediction available")
#     return render(request, 'result.html', {'ans': prediction})

# Create your views here.

# def index(request):
#     return HttpResponse("Hello")

def result(request):
    return render(request,"result.html")

def index(request):
    return render(request,"index.html")

def reg(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request,"Data Inserted Successfully!!")
            return redirect('/login')
    else:
        messages.warning(request,"Please correct the error below!!")
        form = RegistrationForm()
    return render(request,"registration.html", {'form':form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request,"Data Inserted Successfully!!")
                return redirect('predict')
    else:
        form = AuthenticationForm()
    return render(request, "login.html", {'form':form})

from django.shortcuts import render, redirect
from .forms import ContactForm
from django.contrib import messages

def contact_view(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your message has been sent!')
            return redirect('contact')  # Replace with your URL name
    else:
        form = ContactForm()
    return render(request, 'contact.html', {'form': form})



# def predict(request):
#     return render(request,"predict.html")

# def result(request):
#     if request.method == 'POST' and request.FILES.get('image'):
#         img_file = request.FILES['image']  # Get uploaded image

#         # Convert uploaded image to correct format
#         img = image.load_img(img_file, target_size=(224, 224))  # Resize to model input size
#         img_array = image.img_to_array(img) / 255.0  # Normalize pixels (0-1)
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#         # Make prediction
#         prediction = model.predict(img_array)
#         result = "Positive (Cancer Detected)" if prediction[0][0] > 0.5 else "Negative (No Cancer)"

#         return render(request, 'result.html', {
#                         'ans': result,
#                         'title': 'Oral Cancer Prediction',
#         })
    
#     return render(request, 'predict.html', {'title': 'Upload Image'})

