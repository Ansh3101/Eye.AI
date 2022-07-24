from flask import Flask, render_template, request, jsonify
from app.blindness import pred2label, EfficientNetB4, test_transforms
import numpy as np
import torch
from PIL import Image

blindness_weights = 'app/blindness.pth'

app = Flask(__name__) 

@app.route('/', methods=['POST'])
def upload_cufile():
    if request.method == 'POST':
        cu_file = request.files.get('file')
        cu_img = Image.open(cu_file) # Opening The File Using PIL
        cu_img = np.array(cu_img)
        cu_model = EfficientNetB4()
        cu_model.load_state_dict(torch.load(blindness_weights, map_location=torch.device('cpu')))
        cu_model.eval()
        cu_im = test_transforms(cu_img)
        cu_im = torch.reshape(cu_im, (1, 3, 380, 380))
        cu_pred = cu_model(cu_im)
        cu_pred = cu_pred.argmax(dim=-1).numpy()[0]
        cu_pred = pred2label[cu_pred]
        return jsonify({'Prediction': cu_pred})


if __name__ == '__main__':
    app.run()