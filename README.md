# Titanic Survivor Prediction Web App

This is a **Django-based web application** that allows users to interactively predict whether a passenger would have survived the Titanic disaster based on historical passenger data.

The application uses a **custom-built neural network** trained on the Titanic dataset from Kaggle.

---

## Dataset

The dataset used for training the model was sourced from Kaggle:

[Titanic Dataset by yasserh](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

Key features used:

- `Sex` (male/female)
- `Age`
- `SibSp` (number of siblings/spouse aboard)
- `Parch` (number of parents/children aboard)
- `Cabin` (first letter of cabin, missing values treated as "Unknown")

Target column: `Survived` (0 = Did not survive, 1 = Survived)

---

## Model

The model is a **fully connected neural network** implemented from scratch using **NumPy**, without any high-level deep learning library.  

Architecture:

- Input layer: preprocessed features (one-hot encoded categorical + standardized numerical)
- Hidden layers:
  - Layer 1: 64 neurons, ReLU
  - Layer 2: 32 neurons, ReLU
  - Layer 3: 16 neurons, ReLU
- Output layer: 1 neuron, Sigmoid activation for binary classification

**Training details:**

- Loss function: Binary Cross-Entropy
- Optimizer: Gradient Descent
- Learning rate: 0.01
- Epochs: 10,000

Preprocessing includes **one-hot encoding** for categorical variables and **standard scaling** for numerical variables.

The final trained **weights, biases, and preprocessor** are saved as:

- `titanic_weights.npz`
- `Preprocessor.pkl`

These files are used in the Django application for real-time predictions.

---

## Features

- Interactive web form to input passenger details:
  - Gender
  - Age
  - Number of siblings/spouse
  - Number of parents/children
  - Cabin class
- Real-time survival prediction
- Visual feedback:
  - Confetti animation for "Survived"
  - Thematic images for survival/non-survival
- Historical context of the Titanic disaster included on the results page
- Responsive and visually appealing interface

---

## Installation

# 1. Clone the repository
git clone https://github.com/BisheshSubba/Titanic-Survivor.git
cd Titanic-Survivor

# 2. Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Apply Django migrations
python manage.py migrate

# 5. Run the development server
python manage.py runserver
