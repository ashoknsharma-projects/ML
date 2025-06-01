from google.colab import drive
drive.mount('/content/drive')
import tensorflow_hub as hub
import tensorflow as tf
import tf_keras as tfk
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt

def check_gpu():
    print("GPU", "available (YES!!!!)" if tf.config.list_physical_devices("GPU") else "not available :(")

def numeric_sort_key(filename):
    return int(os.path.splitext(filename)[0])
    
### Function to turn Images to Tensors

# Set Image Size
#IMG_SIZE = 224
#COLOR_CHANNELS = 3

def process_image(image_path, img_size, color_channels):
  # Read an image and return tensor of type string
  image = tf.io.read_file(image_path)
  # Convert Tensor string to numerical tensor with 3 color channels (R, G, B)
  image = tf.image.decode_jpeg(image, color_channels)
  # Convert numerical tensor with 3 color channels to a float number between 0 and 1
  image = tf.image.convert_image_dtype(image, tf.float32)
  # Resize the image according to the size set
  image = tf.image.resize(image, size=[img_size, img_size])
  return image

### Function to create Tensor and label as Tuple

def get_image_label(image_path, label, img_size, color_channels):
  image = process_image(image_path, img_size, color_channels)
  return image, label 

# Helper class for mapping process_image
class ImageProcessor:
    def __init__(self, img_size, color_channels):
        self.img_size = img_size
        self.color_channels = color_channels

    def __call__(self, image_path):
        return process_image(image_path, self.img_size, self.color_channels)

# Helper class for mapping get_image_label
class ImageLabelProcessor:
    def __init__(self, img_size, color_channels):
        self.img_size = img_size
        self.color_channels = color_channels

    def __call__(self, image_path, label):
        return get_image_label(image_path, label, self.img_size, self.color_channels)

### Function to create Data into batches

#BATCH_SIZE = 32
def create_data_batches(X, batch_size, y=None, img_size=224, color_channels=3, validation_data=False, test_data=False):
    if test_data:
        print("Creating test data batches...")
        processor = ImageProcessor(img_size, color_channels)
        data = tf.data.Dataset.from_tensor_slices(tf.constant(X))
        data_batch = data.map(processor).batch(batch_size)
        return data_batch

    elif validation_data:
        print("Creating validation data batches...")
        processor = ImageLabelProcessor(img_size, color_channels)
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data_batch = data.map(processor).batch(batch_size)
        return data_batch

    else:
        print("Creating training data batches...")
        processor = ImageLabelProcessor(img_size, color_channels)
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data = data.shuffle(buffer_size=len(X))
        data_batch = data.map(processor).batch(batch_size)
        return data_batch
        
### Load Pretrained model build for our requirements

# INPUT_SHAPE = [batch, height, width, color channels]
#INPUT_SHAPE= [None, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS]
# OUTPUT_SHAPE OF MODEL
#OUTPUT_SHAPE = 2
# MODEL URL
#MODEL_URL = "https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/130-224-classification/2"   
def create_model(input_shape, output_shape, model_url):
  print("Building model with:", model_url)
  # Setup the model layers
  model = tfk.models.Sequential([
  hub.KerasLayer(model_url),
  tfk.layers.Dense(units=output_shape, activation="softmax")])
  model.compile(
      loss=tfk.losses.CategoricalCrossentropy(),
      optimizer=tfk.optimizers.Adam(),
      metrics=["accuracy"])
  model.build(input_shape)
  return model

### Create TensorBoard Callback
def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  return tfk.callbacks.TensorBoard(log_dir)

### Train Model
def train_model(model, train_data_batches, num_epochs, validation_data_batches, early_stopping):
  print("Training model...")
  LOG_DIR_PATH = "drive/MyDrive/Food Classifier/logs"
  tensorboard = create_tensorboard_callback(dir_name=LOG_DIR_PATH,
                                            experiment_name="mobilenet-v2")
  model.fit(train_data_batches,
            epochs=num_epochs,
            validation_data = validation_data_batches,
            validation_freq = 1,
            callbacks=[tensorboard, early_stopping]
            )
  return model

### Function to evaluate prediction probabilities
def get_prediction_label(unique_labels, prediction_probabilities):
  return unique_labels[np.argmax(prediction_probabilities)]

### Function to unbatch
def unbatchify(unique_labels, data):
  images = []
  labels = []
  for image, label in data.unbatch().as_numpy_iterator():
    images.append(image)
    labels.append(unique_labels[np.argmax(label)])
  return images, labels

### Function plotting prediction Probability, Label and Image
def plot_pred(unique_labels, prediction_probabilities, labels, images, n=1):
  """
  View the prediction, ground truth label and image for sample n
  """
  pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]

  # Get the pred label
  pred_label = get_prediction_label(unique_labels,pred_prob)
  # Plot image & remove ticks
  plt.imshow(image)
  plt.xticks([])
  plt.yticks([])
  # Change the color of the title on if the prediction is right or wrong
  if pred_label == true_label:
    color = "green"
  else:
    color = "red"
  # Change plot title to be predicted, probability of prediction and truth label
  plt.title("Pred:{}; Prob:{:2.0f}%; Actual:{}".format(pred_label,
                                   np.max(pred_prob) * 100,
                                   true_label),
                                   color=color) 

### Function to view top 10 predictions
def plot_pred_conf(unique_labels, prediction_probabilities, labels, n=1):
  """
  The top 10 highest prediction confidence along with the true label for the sample n
  """
  pred_prob, true_label = prediction_probabilities[n], labels[n]

  # Get the predicted label
  pred_label = get_prediction_label(unique_labels, pred_prob)

  # Find the top 10 prediction confidence indexes
  top_10_pred_indexes = pred_prob.argsort()[-10:][::-1] # argsort-Gives the order of indexes that will sort an array; [::-1] makes it decending

  # Find the top 10 prediction confidence values
  top_10_pred_values = pred_prob[top_10_pred_indexes]

  # Find the top 10 prediction labels
  top_10_pred_labels = unique_labels[top_10_pred_indexes]

  # Setup plot
  top_plot = plt.bar(np.arange(len(top_10_pred_labels)),
                     top_10_pred_values,
                     color="grey")
  plt.xticks(np.arange(len(top_10_pred_labels)),
             labels=top_10_pred_labels,
             rotation="vertical")

  # Change the color of true label
  if np.isin(true_label, top_10_pred_labels):
    top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")
  else:
    pass

### Save Model
def save_model(dir_name, model, suffix=None):
  """
  Saves a given model in a models directory and appends a suffix (str)
  for clarity
  """
  # Create model directory with current time
  modeldir = dir_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  model_path = modeldir + "-" + suffix + ".h5"
  print(f"Saving model to: {model_path}...")
  model.save(model_path)
  return model_path 
     
### Function to Load Model
def load_model(model_path):
  print(f"Loading saved model from: {model_path}")
  model = tfk.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  return model  

### Train Full Model
def train_full_model(model, full_data_batches, num_epochs, full_model_early_stopping):
  print("Training Full model...")
  LOG_DIR_PATH = "drive/MyDrive/Food Classifier/logs"
  tensorboard = create_tensorboard_callback(dir_name=LOG_DIR_PATH,
                                            experiment_name="mobilenet-v2")
  model.fit(full_data_batches,
            epochs=num_epochs,
            callbacks=[tensorboard, full_model_early_stopping]
            )
  return model