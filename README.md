# Titanic Survivor Prediction Web App

This is a **Django-based web application** that allows users to interactively predict whether a passenger would have survived the Titanic disaster using historical passenger data.

The app uses a **custom-built neural network** trained on the Titanic dataset from Kaggle.

---

## Dataset

Dataset used for training:

[Titanic Dataset by yasserh](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

**Features used:**

- `Sex` (male/female)
- `Age`
- `SibSp` (siblings/spouse aboard)
- `Parch` (parents/children aboard)
- `Cabin` (first letter of cabin, missing values treated as "Unknown")

**Target:** `Survived` (0 = Did not survive, 1 = Survived)

---

## Model

**Fully connected neural network** implemented from scratch using **NumPy**.

**Architecture:**

- Input: preprocessed features (one-hot encoded categorical + standardized numerical)
- Hidden layers:
  - Layer 1: 64 neurons, ReLU
  - Layer 2: 32 neurons, ReLU
  - Layer 3: 16 neurons, ReLU
- Output: 1 neuron, Sigmoid activation

**Training details:**

- Loss: Binary Cross-Entropy
- Optimizer: Gradient Descent
- Learning rate: 0.01
- Epochs: 10,000

Preprocessing includes **one-hot encoding** for categorical variables and **standard scaling** for numerical variables.  
Trained **weights, biases, and preprocessor** are saved as:

- `titanic_weights.npz`
- `Preprocessor.pkl`

These are used in the Django app for real-time predictions.

---

## Features

- Interactive web form for passenger details:
  - Gender, Age, Siblings/Spouse, Parents/Children, Cabin class
- Real-time survival prediction
- Visual feedback:
  - Confetti animation for "Survived"
  - Thematic images for survival/non-survival
- Historical context of the Titanic disaster
- Responsive, visually appealing interface

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/BisheshSubba/Titanic-Survivor.git
cd Titanic-Survivor

# Create & activate virtual environment
python -m venv venv
# Linux/Mac: source venv/bin/activate
# Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Apply migrations and run server
python manage.py migrate
python manage.py runserver
