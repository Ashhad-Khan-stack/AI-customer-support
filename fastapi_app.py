from fastapi import FastAPI
from pydantic import BaseModel
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf

app = FastAPI()

# Paths
MODEL_PATH = "C:/Users/FBC/Desktop/projects/chatbot/ticket_classifier_model"
TOKENIZER_PATH = "C:/Users/FBC/Desktop/projects/chatbot/ticket_classifier_tokenizer"

# Load model + tokenizer
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Define categories & auto-replies
categories = ["Billing", "Technical", "Shipping", "Complaint"]
auto_reply_dict = {
    "Billing": "Your billing issue has been forwarded to our finance team.",
    "Technical": "Our technical team will look into your issue.",
    "Shipping": "Your shipping issue is being processed.",
    "Complaint": "We have received your complaint and will get back soon."
}

# Root endpoint for testing
@app.get("/")
def read_root():
    return {"message": "FastAPI server is running!"}

# Request model
class Ticket(BaseModel):
    text: str

# Predict endpoint
@app.post("/predict")
def predict(ticket: Ticket):
    # Tokenize input
    inputs = tokenizer(ticket.text, return_tensors="tf", truncation=True, padding=True)
    
    # Model prediction
    outputs = model(inputs)
    logits = outputs.logits
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]

    # Map class index to category
    category = categories[predicted_class]
    auto_reply = auto_reply_dict[category]

    # Return both category & auto-reply
    return {"category": category, "auto_reply": auto_reply} 









