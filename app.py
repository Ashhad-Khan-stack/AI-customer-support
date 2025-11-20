# app.py
'''import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification
import numpy as np

# ----------------------------
# 1️⃣ Paths for tokenizer & model
# ----------------------------
TOKENIZER_PATH = r"C:/Users/FBC/Desktop/projects/chatbot/ticket_classifier_tokenizer"
MODEL_PATH = r"C:/Users/FBC/Desktop/projects/chatbot/saved_ticket_classifier_model"

# ----------------------------
# 2️⃣ Load tokenizer & model
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# ----------------------------
# 3️⃣ Define categories (same order as training labels)
# ----------------------------
categories = ["Billing", "Technical", "Shipping", "Complaint"]

# ----------------------------
# 4️⃣ Streamlit UI
# ----------------------------
st.title("Customer Support Ticket Classifier")
st.write("Type your support ticket below and get its predicted category!")

# User input
user_input = st.text_area("Enter ticket text:")

# Predict button
if st.button("Predict Category"):
    if user_input.strip() == "":
        st.warning("Please enter a ticket text!")
    else:
        # ----------------------------
        # 5️⃣ Tokenize input
        # ----------------------------
        inputs = tokenizer(user_input, return_tensors="tf", truncation=True, padding=True)

        # ----------------------------
        # 6️⃣ Model prediction
        # ----------------------------
        outputs = model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
        pred_index = np.argmax(probs)
        pred_category = categories[pred_index]
        confidence = probs[pred_index]

        # ----------------------------
        # 7️⃣ Display result
        # ----------------------------
        st.success(f"Predicted Category: {pred_category}")
        st.info(f"Confidence: {confidence*100:.2f}%")'''

import streamlit as st
import pandas as pd
import requests
import os

# --- Logging module (inline, modular) ---
LOG_FILE = "ticket_history.csv"

def save_ticket_history(results):
    df = pd.DataFrame(results)
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode='a', index=False, header=False)
    else:
        df.to_csv(LOG_FILE, index=False, header=True)

def load_ticket_history():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    else:
        return pd.DataFrame(columns=["Ticket", "Category", "Auto Reply"])

# --- Streamlit UI ---
st.title("AI Customer Support Ticket Classifier")
st.write("Enter multiple tickets (one per line) for batch prediction:")

# Multi-ticket input
tickets_input = st.text_area("Paste 5–10 tickets here:")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict Tickets", "History"])

if page == "Predict Tickets":
    if st.button("Predict Tickets"):
        if not tickets_input.strip():
            st.warning("Please enter at least one ticket!")
        else:
            tickets_list = [t.strip() for t in tickets_input.splitlines() if t.strip()]
            results = []

            url = "http://127.0.0.1:8000/predict"  # FastAPI endpoint

            # Loop through tickets and send to FastAPI
            for ticket in tickets_list:
                payload = {"text": ticket}
                try:
                    response = requests.post(url, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        category = data.get("category", "").replace("\n", " ").strip()
                        auto_reply = data.get("auto_reply", "").replace("\n", " ").strip()
                        results.append({
                            "Ticket": ticket,
                            "Category": category,
                            "Auto Reply": auto_reply
                        })
                    else:
                        results.append({
                            "Ticket": ticket,
                            "Category": "Error",
                            "Auto Reply": "Failed to predict"
                        })
                except Exception as e:
                    results.append({
                        "Ticket": ticket,
                        "Category": "Error",
                        "Auto Reply": str(e)
                    })

            # Display results
            df = pd.DataFrame(results)
            st.table(df)

            st.write("### Prediction Results:")
            for i, row in df.iterrows():
                st.write(f"{i+1}. **Ticket:** {row['Ticket']}")
                st.write(f"   - **Category:** {row['Category']}")
                st.write(f"   - **Auto Reply:** {row['Auto Reply']}")

            # Save to CSV / Logging
            save_ticket_history(results)

            # CSV download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "ticket_predictions.csv", "text/csv")

elif page == "History":
    st.header("Ticket History")
    history_df = load_ticket_history()
    st.dataframe(history_df)

    csv_history = history_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Full History", csv_history, "ticket_history.csv", "text/csv")


