import os
import io,json
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import base64
from flask import Flask,request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt



app = Flask(__name__,)

model = load_model("model/model.h5")




def predictImage(image, model):
    
    
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    result = np.squeeze(model.predict(np.array(image)))

    index = np.argmax(result)

    if result[index] < 0.6:
        return "None Type"

    return "Cat" if index == 0 else "Dog",result[index]


@app.route('/predict',methods=['POST',"GET"])
def predict():
    
    if request.method == "POST":

        encodedImg = request.form.get('file')
        
        imgdata = base64.b64decode(encodedImg)

        imageStream = io.BytesIO(imgdata)
        
        imageFile = Image.open(imageStream)

        imageFile = imageFile.resize((224,224))

        label,confidence =  predictImage((np.asarray(imageFile)),model)
            
        
        return json.dumps([label,str(confidence)])
 
  
    return 'Ok'

        
if __name__ == '__main__':
    
    app.run(debug=True,host='0.0.0.0',port=5000)
