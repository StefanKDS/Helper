import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import tensorflow_hub as hub
from tensorflow.keras import layers
from keras_preprocessing.image import ImageDataGenerator


# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).
    Parameters
    ----------
    filename (str): string filename of target image
    img_shape (int): size to resize target image to, default 224
    scale (bool): whether to scale pixel values to range(0, 1), default True
    """
    # Read in the image
    img = tf.io.read_file(filename)
    # Decode it into a tensor
    img = tf.image.decode_jpeg(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        # Rescale the image (get all values between 0 and 1)
        return img / 255.
    else:
        return img


# Make a function to predict on images and plot them (works with multi-class)
def pred_and_plot(model, filename, class_names, scale=True, shape=224):
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.
    """
    # Import the target image and preprocess it
    img = load_and_prep_image(filename, shape, scale)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    if len(pred[0]) > 1:  # check for multi-class
        pred_class = class_names[pred.argmax()]  # if more than one output, take the max
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]  # if only one output, round

    # Plot the image and predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False);
    plt.show()


def view_random_image(target_dir, target_class):
    # Setup target directory (we'll view images from here)
    target_folder = target_dir + target_class

    # Get a random image path
    random_image = random.sample(os.listdir(target_folder), 1)

    # Read in the image and plot it using matplotlib
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off");
    plt.show();

    print(f"Image shape: {img.shape}")  # show the shape of the image

    return img


def create_model_unfreeze(model_url, num_classes=10, input_shape=(224, 224), unfreeze_layers=0):
    """Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.

    Args:
      model_url (str): A TensorFlow Hub feature extraction URL.
      num_classes (int): Number of output neurons in output layer,
        should be equal to number of target classes, default 10.

    Returns:
      An uncompiled Keras Sequential model with model_url as feature
      extractor layer and Dense output layer with num_classes outputs.
      :param unfreeze_layers:
      :param num_classes:
      :param model_url:
      :param input_shape:
    """
    # Download the pretrained model and save it as a Keras layer
    feature_extractor_layer = hub.KerasLayer(model_url,
                                             trainable=True,  # freeze the underlying patterns
                                             name='feature_extraction_layer',
                                             input_shape=input_shape + (3,))  # define the input image shape

    for layer in feature_extractor_layer[:unfreeze_layers]:
        layer.trainable = False

    # Create our own model
    model = tf.keras.Sequential([
        feature_extractor_layer,  # use the feature extraction layer as the base
        layers.Dense(num_classes, activation='softmax', name='output_layer')  # create our own output layer
    ])

    return model


def create_model(model_url, num_classes=10, input_shape=(224, 224)):
    """Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.

    Args:
      model_url (str): A TensorFlow Hub feature extraction URL.
      num_classes (int): Number of output neurons in output layer,
        should be equal to number of target classes, default 10.

    Returns:
      An uncompiled Keras Sequential model with model_url as feature
      extractor layer and Dense output layer with num_classes outputs.
      :param num_classes:
      :param model_url:
      :param input_shape:
    """
    # Download the pretrained model and save it as a Keras layer
    feature_extractor_layer = hub.KerasLayer(model_url,
                                             trainable=False,  # freeze the underlying patterns
                                             name='feature_extraction_layer',
                                             input_shape=input_shape + (3,))  # define the input image shape

    # Create our own model
    model = tf.keras.Sequential([
        feature_extractor_layer,  # use the feature extraction layer as the base
        layers.Dense(num_classes, activation='softmax', name='output_layer')  # create our own output layer
    ])

    return model


def prepare_augmented_data(train_dir):
    train_datagen_augmented = ImageDataGenerator(rescale=1 / 255.,
                                                 validation_split=0.2,
                                                 rotation_range=0.2,  # rotate the image slightly
                                                 shear_range=0.2,  # shear the image
                                                 zoom_range=0.2,  # zoom into the image
                                                 width_shift_range=0.2,  # shift the image width ways
                                                 height_shift_range=0.2,  # shift the image height ways
                                                 horizontal_flip=True)  # flip the image on the horizontal axis

    train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                       target_size=(224, 224),
                                                                       batch_size=32,
                                                                       class_mode='categorical',
                                                                       shuffle=True,
                                                                       subset='training')

    validation_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                            target_size=(224, 224),
                                                                            batch_size=32,
                                                                            class_mode='categorical',
                                                                            shuffle=True,
                                                                            subset='validation')

    return train_data_augmented, validation_data_augmented


