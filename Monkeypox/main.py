"""

The 7 classes of skin cancer lesions included in this dataset are:
Melanocytic nevi (nv)
Melanoma (mel)
Benign keratosis-like lesions (bkl)
Basal cell carcinoma (bcc)
Actinic keratoses (akiec)
Vascular lesions (vas)
Dermatofibroma (df)

"""

import numpy as np
import sklearn
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


def getPrediction(filename):
    classes = ['Monkeypox', 'Others']

    # Load model
    my_model = load_model("/ResModel23082022_v3_0.93")

    SIZE = 224  # Resize to same size as training images
    img_path = 'static/images/' + filename
    img = np.asarray(Image.open(img_path).resize((SIZE, SIZE)))

    img = img / 255.  # Scale pixel values

    pred = my_model.predict(img)  # Predict

    # Convert prediction to class name
    pred_class = le.inverse_transform([np.argmax(pred)])[0]
    print("Diagnosis is:", pred_class)
    return pred_class

# a =getPrediction('vasc.jpg')