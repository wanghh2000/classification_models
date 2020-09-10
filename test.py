import numpy as np
from skimage.io import imread
from skimage.transform import resize
from keras.applications.imagenet_utils import decode_predictions
from classification_models.keras import Classifiers

ResNet18, preprocess_input = Classifiers.get('resnet18')

# read and prepare image
#x = imread('./imgs/tests/seagull.jpg')
x = imread('./tests/data/dog.jpg')
x = resize(x, (224, 224)) * 255    # cast back to 0-255 range
x = preprocess_input(x)
x = np.expand_dims(x, 0)

# load model
model = ResNet18(input_shape=(224,224,3), weights='imagenet', classes=1000)

# processing image
y = model.predict(x)

# result
print(decode_predictions(y))