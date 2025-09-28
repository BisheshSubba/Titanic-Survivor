from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import pandas as pd
import pickle
import os


def home(request):
    return render(request, "landing/main.html")

# Load preprocessor
preprocessor_path = os.path.join(os.path.dirname(__file__), "Preprocessor.pkl")
with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)

# Load weights
weights_path = os.path.join(os.path.dirname(__file__), "titanic_weights.npz")
data = np.load(weights_path)
w1, b1 = data['w1'], data['ba1']
w2, b2 = data['w2'], data['ba2']
w3, b3 = data['w3'], data['ba3']
w4, b4 = data['w4'], data['ba4']

# Activation functions
def relu(X): return np.maximum(0, X)
def sigmoid(X): return 1 / (1 + np.exp(-X))

# Prediction function
def predict_survival(sex, age, sibsp, parch, cabin):
    # Construct input dataframe
    input_df = pd.DataFrame([[sex, age, sibsp, parch, cabin]],
                            columns=['Sex', 'Age', 'SibSp', 'Parch', 'Cabin'])

    # Apply preprocessing
    final_data = preprocessor.transform(input_df)

    # Forward pass
    z1 = relu(final_data @ w1 + b1)
    z2 = relu(z1 @ w2 + b2)
    z3 = relu(z2 @ w3 + b3)
    z4 = sigmoid(z3 @ w4 + b4)

    # Return binary prediction
    return int((z4 > 0.5).astype(int)[0][0])

# API endpoint
@csrf_exempt
def survival_api(request):
    if request.method == "POST":
        sex = request.POST.get("gender", "unknown")
        age = float(request.POST.get("age", 0))
        sibsp = int(request.POST.get("sibsp", 0))
        parch = int(request.POST.get("parch", 0))
        cabin = request.POST.get("cabin", "U")

        prediction = predict_survival(sex, age, sibsp, parch, cabin)
        result_text = "Survived" if prediction == 1 else "Did not survive"

        return render(request, "landing/result.html", {"prediction": result_text})

    return JsonResponse({"error": "POST request required"})
