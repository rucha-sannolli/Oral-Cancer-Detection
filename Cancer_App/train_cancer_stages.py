import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import joblib  # For saving scaler and model

# Load the dataset
file_path = "D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/Cancer_App/dataset/synthetic_cancer_stages_dataset.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Display dataset preview
print("Original Dataset Preview:")
print(df.head())

# Check and handle missing values (imputation)
print("\nMissing Values Before Handling:")
print(df.isnull().sum())

imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Re-check for any missing values
print("\nMissing Values After Handling:")
print(df_imputed.isnull().sum())

# # Encode categorical features using one-hot encoding -------
# df_encoded = pd.get_dummies(df_imputed, drop_first=True) -----

# # Define features (X) and target (y) ---------
# X = df_encoded.drop('Cancer Stage', axis=1, errors='ignore')  # Features -------
# y = df_imputed['Cancer Stage']  # Target with all stages (Stage I, II, III, IV) ------


# Separate features and target first
X_raw = df_imputed.drop('Cancer Stage', axis=1)
y = df_imputed['Cancer Stage']

# One-hot encode only the features (not target!)
X_encoded = pd.get_dummies(X_raw, drop_first=False)


# Apply SMOTE for class balancing
# ➤ Ensure all features are numeric (important for SMOTE)
# X = X.astype(float) ----
X = X_encoded.astype(float)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check class distribution after balancing
print("\nBalanced Stage Counts:")
print(y_resampled.value_counts())

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the fitted scaler for future use
joblib.dump(scaler, 'scaler.pkl')

# Train an SVM model with RBF kernel for better handling of non-linear relationships
svm_model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
svm_model.fit(X_train, y_train)

# Save the trained SVM model
joblib.dump(svm_model, 'svm_model_multi_class_rbf.pkl')

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
stages = ['Stage 0', 'Stage I', 'Stage II', 'Stage III', 'Stage IV']
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=stages, yticklabels=stages)
plt.xlabel("Predicted Stage")
plt.ylabel("Actual Stage")
plt.title("Confusion Matrix - Multi-class SVM")
plt.show()

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the predicted labels for analysis (optional)
df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_results.to_csv('cancer_stage_predictions.csv', index=False)
print("\nSaved predictions to 'cancer_stage_predictions.csv'")

# Save the feature names after encoding for alignment later
joblib.dump(X_encoded.columns, 'feature_names.pkl')

# Check the saved feature names
feature_names = joblib.load('feature_names.pkl')
print("\nFeature Names:")
for feature in feature_names:
    print(feature)






    



#RANDOM FOREST
# # import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# from imblearn.over_sampling import SMOTE
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib

# # Load the dataset
# file_path = "D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/Cancer_App/dataset/synthetic_cancer_stages_dataset.csv"
# df = pd.read_csv(file_path)

# # Handle missing values
# imputer = SimpleImputer(strategy='most_frequent')
# df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# # Separate features and target
# X_raw = df_imputed.drop('Cancer Stage', axis=1)
# y = df_imputed['Cancer Stage']

# # One-hot encode categorical features
# X_encoded = pd.get_dummies(X_raw, drop_first=False)
# X = X_encoded.astype(float)

# # Apply SMOTE for balancing
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# # Scale features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# joblib.dump(scaler, 'scaler_rf.pkl')

# # Train Random Forest
# rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
# rf_model.fit(X_train_scaled, y_train)
# joblib.dump(rf_model, 'rf_model.pkl')

# # Predict
# y_pred_rf = rf_model.predict(X_test_scaled)

# # Evaluate
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# print(f"\nRandom Forest Accuracy: {accuracy_rf * 100:.2f}%")

# print("\nClassification Report (RF):")
# print(classification_report(y_test, y_pred_rf))

# # Confusion Matrix
# cm_rf = confusion_matrix(y_test, y_pred_rf)
# stages = ['Stage 0', 'Stage I', 'Stage II', 'Stage III', 'Stage IV']
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", xticklabels=stages, yticklabels=stages)
# plt.xlabel("Predicted Stage")
# plt.ylabel("Actual Stage")
# plt.title("Confusion Matrix - Random Forest")
# plt.show()

# # Save predictions
# pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf}).to_csv('rf_cancer_stage_predictions.csv', index=False)

# # Feature Importance
# importances = rf_model.feature_importances_
# features = X_encoded.columns
# indices = np.argsort(importances)[::-1]

# plt.figure(figsize=(10, 8))
# sns.barplot(x=importances[indices][:15], y=features[indices][:15])
# plt.title("Top 15 Feature Importances - Random Forest")
# plt.tight_layout()
# plt.show()

# # Save feature names for alignment
# joblib.dump(X_encoded.columns, 'feature_names_rf.pkl')
#RANDOM FOREST




























# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib  # For saving scaler and model

# # Load the dataset
# file_path = "D:/AI DS/BE/Final year project 8th sem/Oral Cancer/Cancer_Project/Cancer_App/dataset/synthetic_cancer_stages_dataset.csv"
# df = pd.read_csv(file_path)

# # Display first few rows to confirm the structure
# print("Dataset preview:")
# print(df.head())

# # Check for missing values
# print("\nMissing Values:")
# print(df.isnull().sum())

# # Handle missing values (imputation with most frequent values)
# imputer = SimpleImputer(strategy='most_frequent')
# df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# # Encode categorical features using one-hot encoding
# df_encoded = pd.get_dummies(df_imputed, drop_first=True)

# # Define features (X) and target (y)
# X = df_encoded.drop('Cancer Stage_Stage IV', axis=1, errors='ignore')  # Drop target if it's present in features
# y = df_encoded['Cancer Stage_Stage IV']

# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Save the fitted scaler for later use in predictions
# joblib.dump(scaler, 'scaler.pkl')

# # Train an SVM with class weights to handle imbalance
# svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
# svm_model.fit(X_train, y_train)

# # Save the trained SVM model
# joblib.dump(svm_model, 'svm_model.pkl')

# # Predict on the test set
# y_pred = svm_model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='binary', zero_division=1)
# recall = recall_score(y_test, y_pred, average='binary', zero_division=1)
# f1 = f1_score(y_test, y_pred, average='binary')

# print(f"\nModel Evaluation Metrics:")
# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1 Score: {f1:.2f}")

# # Generate and print confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# print("\nConfusion Matrix:")
# print(cm)

# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Cancer Stage IV', 'Cancer Stage IV'],
#             yticklabels=['No Cancer Stage IV', 'Cancer Stage IV'])
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()

# # Classification report
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # Feature importance estimation using SVM's coefficients
# coefficients = np.abs(svm_model.coef_[0])  # Absolute value of coefficients for interpretability
# feature_importance = pd.Series(coefficients, index=X.columns).sort_values(ascending=False)

# # Display top features
# print("\nTop 10 Important Features Based on Coefficients:")
# print(feature_importance.head(10))

# # Plot feature importance
# plt.figure(figsize=(10, 6))
# feature_importance.head(10).plot(kind='bar')
# plt.title("Top 10 Feature Importance (SVM Coefficients)")
# plt.xlabel("Features")
# plt.ylabel("Coefficient Magnitude")
# plt.xticks(rotation=45)
# plt.show()
