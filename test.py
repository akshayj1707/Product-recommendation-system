import pickle
import tensorflow
import numpy as np
from PIL import Image
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2
import faiss
import boto3
import io

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

s3 = boto3.client('s3')
s3 = boto3.resource(
    service_name='s3',
    region_name='ap-south-1',
    aws_access_key_id='AKIAWM5ZNPZ2DYQVM5HM',
    aws_secret_access_key='hedveM0xy3vv7iFoayK54eh58tpNCruQZmTQt7Sv'
)


#img = image.load_img('D:\PYTHON\Data science\projects\my projects\product recommender system\prs\sample1.jpg',target_size=(224,224))
# Specify the S3 bucket name and image file key
bucket_name = 'akshaytest'
image_key = 'download.jpg'

# Download the image from S3
obj = s3.Bucket('akshaytest').Object('download.jpg').get()
image_data = obj['Body'].read()

# Create an in-memory file-like object
image_stream = io.BytesIO(image_data)

# Read the image using Pillow (PIL) library
image = Image.open(image_stream)
img = image.resize((224,224))
#img = image.load_img(obj,target_size=(224,224))
img_array = img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

#neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
#neighbors.fit(feature_list)

#distances,indices = neighbors.kneighbors([normalized_result])

k = 5
faiss_index = faiss.IndexFlatL2(2048)        # build the index, need to input embedding size (last layer dimension of our model)
print(faiss_index.is_trained)

# adding the index embeddings to faiss
faiss_index.add(np.array(feature_list))

# check how many are added
print("total embeddings added", faiss_index.ntotal) 
_, indices = faiss_index.search(np.array([normalized_result]),5)


print(indices)

for file in indices[0][1:6]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    cv2.waitKey(0)