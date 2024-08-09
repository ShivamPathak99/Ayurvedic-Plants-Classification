import streamlit as st # For UI Interface 
from fastcore.all import * # Dependencies 
from fastai.vision.all import * # Machine Learing Library 
from pathlib import Path
from fastai.vision.all import *
from fastai.text.all import *
from fastai.collab import *
from fastai.tabular.all import * 
import requests # 
from g4f.client import Client  # Gpt Client GPT 3.5 Turbo 

# Set page configuration
st.set_page_config(layout="wide", page_title="Ayurvedic Plant Classifier")

# Define the path to your model file
model_path = Path('plant_model.pkl')  # Improting model from its directory pwd

# Load the model using the Path object
learn_inf = load_learner(model_path)

# Function to classify the image
def classify_img(data):
    pred, pred_idx, probs = learn_inf.predict(data)
    return pred, probs[pred_idx]

# Streamlit app layout
st.title("Medicinal Plant Classifier 🌿🏵️🌱")
st.write("Upload an image of a plant to identify it and learn about its health benefits.")


# Sidebar for uploading and downloading
st.sidebar.write("## Upload and Classify :gear:")
uploaded_image = st.sidebar.file_uploader("Upload an image of a plant", type=["png", "jpg", "jpeg"])
st.sidebar.write('')
if uploaded_image:
    bytes_data = uploaded_image.getvalue()
    # Display the uploaded image with a fixed width
    col1, col2 = st.columns(2)

    classify = st.sidebar.button("Classify the Plant ☘️")
    if classify:
        # Add a spinner for the classifier prediction
        label, confidence = classify_img(bytes_data)


        # Use Gemini API to get health benefits description
        query = f"Describe  the top 5 health benefits of {label.lower()} as a medicinal plant [Answer in short 5 points], answer with reference to ayurveda (provide weblinks more more info), use emojis also"
        client = Client()
        
        # Fetch health benefits without displaying a progress bar
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query}]
        )
        
        with col1:
            st.image(bytes_data, caption="Uploaded image", width=400) # Set the width to 400 pixels
            st.write(f"It is - **{label}**!    (Confidence **{confidence * 100:.02f}%**)")

        health_benefits = response.choices[0].message.content
        
        # Create a column layout for the image and health benefits
           
        with col2:
            st.subheader(f"Health Benefits of {label}:")
            st.write(health_benefits.strip()) # Display the health benefits in the second column
