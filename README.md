# 🚀 Spam Detection API (FastAPI)

A REST API for detecting whether a message is **Spam or Ham (Not Spam)** using a Machine Learning model.

This API serves predictions from a trained TF-IDF + XGBoost pipeline and is designed to be consumed by frontend apps like Streamlit.

---

# 🎯 Features

✅ Spam/Ham classification  
✅ Confidence score output  
✅ FastAPI-powered REST API  
✅ Pre-trained ML pipeline  
✅ Swagger UI docs  
✅ Ready for cloud deployment  

---

# 🧠 Model Overview

### Text Processing
- Lowercasing  
- URL & special character removal  
- Whitespace normalization  

### Feature Engineering
- TF-IDF Vectorization  
- Unigrams + Bigrams  

### Model
- XGBoost Classifier  
- Implemented via Scikit-learn Pipeline  

---

# 📂 Project Structure

```

fastapi-spam-api/
│
├── main.py            # FastAPI app
├── model.pkl          # Trained ML pipeline
├── requirements.txt
└── README.md

```

---

# ⚙️ Installation & Local Run

## 1️⃣ Clone Repository

```

git clone https://github.com/Balusanu/Spam-detection--FastAPI
cd fastapi-spam-api

```

---

## 2️⃣ Install Dependencies

```

pip install -r requirements.txt

```

---

## 3️⃣ Run API Server

```

uvicorn main:app --reload

```

Server runs at:

```

[http://127.0.0.1:8000](http://127.0.0.1:8000)

```

---

# 📘 API Usage

## 🔹 Health Check

### GET /

```

{
"status": "API running"
}

````

---

## 🔹 Predict Spam

### POST /predict

### Request Body

```json
{
  "message": "Congratulations! You won a free iPhone."
}
````

### Response

```json
{
  "prediction": "Spam",
  "confidence": 0.97
}
```

---

# 📄 API Docs

Interactive Swagger docs:

```
/docs
```

Example:

```
http://127.0.0.1:8000/docs
```

---

# ☁️ Deployment

This API can be deployed on:

* Render
* Railway
* Fly.io
* Docker containers

Example start command:

```
uvicorn main:app --host 0.0.0.0 --port 10000
```

---

# ⚠️ Limitations

* Model trained on older SMS/email spam dataset
* May not fully detect modern phishing styles
* Requires periodic retraining for production use

---

# 🔮 Future Improvements

* Transformer-based models (DistilBERT)
* URL/domain reputation features
* Email header analysis
* Logging & monitoring
* Auto retraining pipeline

---

# 🛠 Tech Stack

* Python
* FastAPI
* Scikit-learn
* XGBoost
* Uvicorn

---

# 👨‍💻 Author

**Balasubramanya C K**