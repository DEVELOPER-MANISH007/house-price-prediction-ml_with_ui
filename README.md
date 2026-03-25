# 🏠 House Price Prediction Web App

An end-to-end Machine Learning project that predicts house prices using a trained model and an interactive web interface built with Streamlit.

---

## 🚀 Features

* 📊 Data preprocessing using Scikit-learn Pipeline
* 🧠 Model training using Random Forest Regressor
* 📦 Model & pipeline saving using Joblib
* 💻 Command Line Interface (CLI) for training & prediction
* 🌐 Interactive Web App using Streamlit
* ⚡ Clean and modular project structure

---

## 🧠 Tech Stack

### 🔹 Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn

### 🔹 Deployment / UI

* Streamlit

### 🔹 Tools

* Joblib
* Git & GitHub

---

## 📂 Project Structure

```bash
project_working/
│
├── data/
│   └── 05_housing.csv
│
├── models/
│   ├── model.pkl
│   └── pipeline.pkl
│
├── src/
│   ├── main.py        # CLI (train & predict)
│   └── app.py         # Streamlit UI
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/DEVELOPER-MANISH007/your-repo-name.git
cd your-repo-name
```

---

### 2️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

```bash
python src/main.py train
```

👉 This will:

* Train the model
* Save model & pipeline in `/models`

---

## 🔮 Run the Web App

```bash
streamlit run src/app.py
```

👉 Open in browser → enter inputs → get predictions

---

## 💻 CLI Prediction (Optional)

```bash
python src/main.py predict
```

---

## 📸 Screenshots

👉 Add your Streamlit UI screenshots here

---

## 🎯 Key Concepts Used

* Data Cleaning & Preprocessing
* Feature Scaling & Encoding
* Pipeline Automation
* Model Training & Evaluation
* Model Persistence
* UI Integration

---

## 🚀 Future Improvements

* Hyperparameter tuning
* Model comparison
* Better UI/UX
* Deployment (Streamlit Cloud / AWS)

---

## 🧑‍💻 Author

**Manish Kumar**
📧 [manish8755026341@gmail.com](mailto:manish8755026341@gmail.com)
🔗 GitHub: https://github.com/DEVELOPER-MANISH007

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
