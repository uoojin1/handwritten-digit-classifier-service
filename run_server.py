# import packages
from keras.models import Sequential
from keras.layers import Dense, Lambda, Flatten, Convolution2D, MaxPooling2D, SpatialDropout2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras_preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
from flask_cors import CORS
import io
from w3lib.url import parse_data_uri

# initialize the flask application and the keras model
app = flask.Flask(__name__)
CORS(app)

model = None

def load_model():
  global model
  # load the pretrained imagenet weight
  # create a sequential model
  model = Sequential()

  # standardize and flatten the 28 x 28 input
  # element-wise standardization
  model.add(Convolution2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
  model.add(Convolution2D(32, (3,3), activation='relu'))
  model.add(MaxPooling2D())
  model.add(Convolution2D(32, (3,3), activation='relu'))
  model.add(Convolution2D(32, (3,3), activation='relu'))
  model.add(MaxPooling2D())
  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(10, activation='softmax'))

  rms = RMSprop(learning_rate=0.001, rho=0.9)

  model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy'])

  model.load_weights('my_model_weights.h5')

def prepare_image(image, targetSize):
  '''
  Function that prepares the given image into a consumable state for the ML model.
  Parameters:
    image - the input image that will be processed
    targetSize - the size that the processed image should be. ex. (244, 244)
  '''
  # convert image to Grayscale
  if image.mode != "L":
    image = image.convert('RGB')
    image = image.convert('L')

  # resize the image 
  image = image.resize(targetSize)
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)

  # return the preprocessed image
  return image

@app.route("/predict", methods=["POST"])
def predict():
  data = {"success": False}

  if flask.request.method == "POST":
    imgData = flask.request.data
    parsed = parse_data_uri(imgData)

    if parsed:
      # read the image in PIL format
      image = Image.open(io.BytesIO(parsed.data))
      # preprocess the image prior to classification
      image = prepare_image(image, (28, 28))
      # classify the input image with the model
      preds = model.predict(image)
      # decode the prediction results
      label = int(np.argmax(preds))
      probability = preds[0][label] * 100
      # form the response
      result = { "label": label, "probability": float(probability) }
      data["prediction"] = result
      data["success"] = True

  return flask.jsonify(data)

if __name__ == "__main__":
  print("Starting the flask server...")
  port = int(os.environ.get("PORT", 5000))
  print("port", port)
  load_model()
  app.run(threaded=False, port=port)