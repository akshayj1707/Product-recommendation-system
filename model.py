import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
import boto3

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#print(model.summary())

def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

#filenames = []

#for file in os.listdir('D:\PYTHON\Data science\projects\my projects\product recommender system\prs\images'):
#    filenames.append(os.path.join('D:\PYTHON\Data science\projects\my projects\product recommender system\prs\images',file))

s3 = boto3.client('s3')
s3 = boto3.resource(
    service_name='s3',
    region_name='ap-south-1',
    aws_access_key_id='AKIAWM5ZNPZ2DYQVM5HM',
    aws_secret_access_key='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'
)

filenames = []

s3 = boto3.client('s3')

s3 = boto3.resource(
    service_name='s3',
    region_name='ap-south-1',
    aws_access_key_id='AKIAWM5ZNPZ2DYQVM5HM',
    aws_secret_access_key='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'
)


my_bucket = s3.Bucket('akshaytest')

filenames = []
for file in my_bucket.objects.filter(Prefix="images/"):
    file_name=file.key
    if file_name.find(".jpg")!=-1:
        filenames.append(file.key)

feature_list = []

#for file in tqdm(filenames):
#    feature_list.append(extract_features(file,model))

for file in tqdm(my_bucket.objects.filter(Prefix="images/")):
    file_name=file.key
    if file_name.find(".jpg")!=-1:
        feature_list.append(extract_features(file.key,model))

pickle.dump(feature_list,open('s3embeddings.pkl','wb'))
pickle.dump(filenames,open('s3filenames.pkl','wb'))

