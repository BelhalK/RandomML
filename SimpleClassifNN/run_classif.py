#python3 run_classif.py --model=modelname --imagepath=path/to/image


from matplotlib import pyplot as plt
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import cv2

import argparse



ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,help="")
ap.add_argument("-i", "--imagepath", type=str,help="")
args = vars(ap.parse_args())


MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception, # TensorFlow ONLY
    "resnet": ResNet50
}


modelname = args["model"]

inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input
if modelname in ("inception", "xception"):
    inputShape = (299, 299)
    preprocess = preprocess_input


print("[INFO] loading {}...".format(modelname))
Network = MODELS[modelname]
model = Network(weights="imagenet")


#imagepath = "images/chelsea.png"
imagepath = args["imagepath"]
print("[INFO] loading and pre-processing image...")
image = load_img(imagepath, target_size=inputShape)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess(image)


print("[INFO] classifying image with '{}'...".format(model))
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))



# Load an color image in grayscale
img = cv2.imread(imagepath,1)
(imagenetID, label, prob) = P[0][0]
cv2.putText(img, "Label: {}, {:.2f}%".format(label, prob * 100),(10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imwrite("output.png", img)