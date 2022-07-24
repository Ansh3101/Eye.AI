from flask import Flask, render_template, request, jsonify
from app.eyedisease import ed_pred2label, EfficientNetB3, ed_test_transforms
import numpy as np
import torch
from PIL import Image

eyedisease_weights = 'app/eye-disease.pth'

app = Flask(__name__) 

@app.route('/', methods=['POST'])
def upload_edfile():
    if request.method == 'POST':
        ed_file = request.files.get('file')
        ed_img = Image.open(ed_file) # Opening The File Using PIL
        ed_img = np.array(ed_img)
        ed_model = EfficientNetB3()
        ed_model.load_state_dict(torch.load(eyedisease_weights, map_location=torch.device('cpu')))
        ed_model.eval()
        ed_im = ed_test_transforms(ed_img)
        ed_im = torch.reshape(ed_im, (1, 3, 300, 300))
        ed_pred = ed_model(ed_im)
        ed_pred = ed_pred.argmax(dim=-1).numpy()[0]
        ed_pred = ed_pred2label[ed_pred]
        return jsonify({'Prediction': ed_pred})


if __name__ == '__main__':
    app.run()