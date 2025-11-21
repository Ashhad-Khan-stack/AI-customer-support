# AI-customer-support
AI Customer Support 🤖💬

🌟 Project Overview

AI Customer Support is a smart chatbot system designed to automate customer support tickets using AI and NLP.
It can understand user queries, classify ticket types, and maintain conversation history, providing a faster and more efficient support experience.

⚡ Features

Automatic ticket classification using a trained TensorFlow model

FastAPI backend for API endpoints

Interactive chat interface for customer support

Supports ticket history logging

Easily deployable on local or cloud server

🧰 Tech Stack

Python 3.11

FastAPI for backend APIs

TensorFlow for AI model

Pandas & CSV for ticket history management

Git LFS for large model files

🛠 Setup & Installation

Clone the repository:

git clone https://github.com/Ashhad-Khan-stack/AI-customer-support.git
cd AI-customer-support

Create virtual environment & install dependencies:

python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt

Run the FastAPI server:

uvicorn app:app --reload

Access the API or chat interface at:

http://127.0.0.1:8000
📝 Model Info

Ticket Classifier Model: ticket_classifier_model/tf_model.h5

Tokenizer & preprocessing: stored in ticket_classifier_tokenizer/

LFS tracked model for smooth GitHub storage

💡 Usage Example
import requests

url = "http://127.0.0.1:8000/predict_ticket"
data = {"query": "My order hasn't arrived yet!"}
response = requests.post(url, json=data)
print(response.json())
📂 Project Structure
AI-customer-support/
├─ app.py
├─ fastapi_app.py
├─ ticket_classifier_model/
├─ ticket_classifier_tokenizer/
├─ saved_ticket_classifier_model/
├─ tickets.csv
├─ ticket_history.csv
├─ README.md
└─ .gitattributes
🎯 Future Enhancements

Integrate GUI/Frontend interface for end-users

Add multi-language support for tickets

Deploy to cloud services (AWS, GCP, Azure)

📌 Author

Muhammad Ashhad U Rehman Khan
LinkedIn | GitHub
