import streamlit as st
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import boto3

# S3 config
bucket_name = 'ml-app-bucket-ridwan'
local_model_dir = 'imdb_sent_model'
s3_prefix = 'imdb_sent_model/'

# S3 downloader
s3 = boto3.client('s3')

def download_model_from_s3(local_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for obj in result['Contents']:
                s3_key = obj['Key']
                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                s3.download_file(bucket_name, s3_key, local_file)

# UI
st.title("🧠 IMDB Sentiment Predictor (from S3 model)")

# Download model
if st.button("⬇️ Download Model from S3"):
    with st.spinner("Downloading model files..."):
        download_model_from_s3(local_model_dir, s3_prefix)
    st.success("✅ Model downloaded!")

# Text input
text = st.text_area("✍️ Enter your movie review:", "This movie was absolutely amazing!")

# Predict
if st.button("🔍 Predict Sentiment"):
    with st.spinner("Loading model & predicting..."):
        # Load model from local folder
        model = AutoModelForSequenceClassification.from_pretrained(local_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

        classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)
        result = classifier(text)[0]
        st.success(f"**Prediction:** {result['label']} ({result['score']:.2f})")
