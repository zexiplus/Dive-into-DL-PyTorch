# requirements

import io
import json
import os

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

# pre-processing

# transform image
def transform_image(infile):
    input_transforms = [transforms.Resize(255), # resize to 255, 255
        transforms.CenterCrop(224), # center 
        transforms.ToTensor(), # to tensor
        # 
        transforms.Normalize([0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])]
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(infile)
    timg = my_transforms(image)
    timg.unsqueeze_(0) # [1,2,3] -> [[1, 2, 3]] 因为预测函数输入是批量个 tensor
    return timg

# predict
def get_prediction(input_tensor):
    outputs = model.forward(input_tensor)
    _, y_hat = outputs.max(1)
    prediction = y_hat.item()
    return prediction

# get prediction label
def render_prediction(prediction_idx):
    stridx = str(prediction_idx)
    class_name = 'Unknown'
    if img_class_map is not None:
        if stridx in img_class_map is not None:
            class_name = img_class_map[stridx][1]

    return prediction_idx, class_name

# create Flask app
app = Flask(__name__)

 # Trained on 1000 classes from ImageNet
model = models.densenet121(pretrained=True)              
model.eval()

img_class_map = None
mapping_file_path = './data/index_to_name.json'

# read file
if os.path.isfile(mapping_file_path):
    with open (mapping_file_path) as f:
        img_class_map = json.load(f)


@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'Try POSTing to the /predict endpoint with an RGB image attachment'})

# accept redict request
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            input_tensor = transform_image(file)
            prediction_idx = get_prediction(input_tensor)
            class_id, class_name = render_prediction(prediction_idx)
            return jsonify({'class_id': class_id, 'class_name': class_name})

# start up server
if __name__ == '__main__':
    app.run()


#### start server
# FLASK_APP=app.py flask run

#### request
# curl -X POST -H "Content-Type: multipart/form-data" http://localhost:5000/predict -F "file=@../data/kitten.jpg"
