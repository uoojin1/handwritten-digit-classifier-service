# import packages
from keras.applications import ResNet50
from keras_preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

# initialize the flask application and the keras model
app = flask.Flask(__name__)
model = None

def load_model():
  global model
  # load the pretrained imagenet weight
  model = ResNet50(weights="imagenet")

def prepare_image(image, targetSize):
  '''
  Function that prepares the given image into a consumable state for the ML model.
  Parameters:
    image - the input image that will be processed
    targetSize - the size that the processed image should be. ex. (244, 244)
  '''
  # convert image to RGB
  if image.mode != "RGB":
    image = image.convert("RGB")
  print(image)
  # resize the image 
  image = image.resize(targetSize)
  # transform the image into a 3D tensor for the preprocessor
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  # preprocess the image via mean subtraction and scaling
  image = imagenet_utils.preprocess_input(image)

  # return the preprocessed image
  return image

@app.route("/predict", methods=["POST"])
def predict():
  data = {"success": False}

  if flask.request.method == "POST":
    if flask.request.files.get("image"):
      # read the image in PIL format
      image = flask.request.files["image"].read()
      image = Image.open(io.BytesIO(image))

      # preprocess the image prior to classification
      image = prepare_image(image, (224, 224))

      # classify the input image with the model
      preds = model.predict(image)

      # decode the prediction results
      results = imagenet_utils.decode_predictions(preds)
      data["predictions"] = []

      # loop over the results and add them to the list of predictions
      for (_, label, probability) in results[0]:
        result = { "label": label, "probability": float(probability) }
        data["predictions"].append(result)
      
      data["success"] = True

  return flask.jsonify(data)

if __name__ == "__main__":
  print "Starting the flask server..."
  load_model()
  app.run(threaded=False)