import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2
import faiss
import boto3
import io

feature_list = np.array(pickle.load(open('s3embeddings.pkl','rb')))
filenames = pickle.load(open('s3filenames.pkl','rb'))
    
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Product Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    k = 5
    faiss_index = faiss.IndexFlatL2(2048)        # build the index, need to input embedding size (last layer dimension of our model)
    print(faiss_index.is_trained)

    # adding the index embeddings to faiss
    faiss_index.add(np.array(feature_list))

    # check how many are added
    print("total embeddings added", faiss_index.ntotal) 
    _, indices = faiss_index.search(np.array([features]),k)

    
    return indices

# steps
# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # feature extract
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
        # recommendention
        indices = recommend(features,feature_list)

        # Initialize the S3 client
        s3 = boto3.client('s3')

        s3 = boto3.resource(
            service_name='s3',
            region_name='ap-south-1',
            aws_access_key_id='AKIAWM5ZNPZ2DYQVM5HM',
            aws_secret_access_key='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'
        )

        # Specify the S3 bucket name 
        my_bucket = s3.Bucket('akshaytest')

        # show
        labels = ['col1','col2','col3','col4','col5']

        cols = st.columns(len(labels))
        for i in labels:
            with cols[labels.index(i)]:
                obj = my_bucket.Object(filenames[indices[0][labels.index(i)]]).get()
                image_data = obj['Body'].read()
                # Create an in-memory file-like object
                image_stream = io.BytesIO(image_data)
                # Read the image using Pillow (PIL) library
                image = Image.open(image_stream)
                st.image(image)


    else:
        st.header("Some error occured in file upload")
