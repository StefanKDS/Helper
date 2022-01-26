#############################################################
# Load data
#############################################################

from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from numpy import random
import tensorflow as tf
from helper_functions import view_random_image, walk_through_dir, plot_loss_curves

# train
# |
# +-- cats
# |     +- 1.jpg
# |     +- 2.jpg
# |     +- ....
# |
# +-- dogs
#       +- 1.jpg
#       +- 2.jpg
#       +- ....

# Verzeichnisse anschauen
walk_through_dir("train")

# Get the class names (programmatically, this is much more helpful with a longer list of classes)
import pathlib
import numpy as np

train_dir = pathlib.Path("train/")  # turn our training path into a Python path
class_names = np.array(
    sorted([item.name for item in train_dir.glob('*')]))  # created a list of class_names from the subdirectories
print(class_names)

# Show a random picture of each class
img = view_random_image(target_dir="train/",
                        target_class="cats")

img = view_random_image(target_dir="train/",
                        target_class="dogs")

# Prepare training data
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
                                                                   class_mode='binary',
                                                                   shuffle=True,
                                                                   subset='training')

validation_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                        target_size=(224, 224),
                                                                        batch_size=32,
                                                                        class_mode='binary',
                                                                        shuffle=True,
                                                                        subset='validation')

# Visualize
augmented_images, augmented_labels = train_data_augmented.next()
# Show original image and augmented image
random_number = random.randint(0, 32)  # we're making batches of size 32, so we'll get a random instance
img = augmented_images[random_number]
plt.imshow(img)
plt.title(f"Augmented image")
plt.axis(False);
plt.show();
print(f"Image shape: {img.shape}")

#############################################################
# CNN Model
#############################################################

# Create model
model_1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),  # same input shape as our images
    tf.keras.layers.Conv2D(10, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
 #   tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(10, 3, activation='relu'),
    tf.keras.layers.Conv2D(10, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
  #  tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Create callbacks
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                       patience=3, min_lr=0.001)

earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

checkpoint_filepath = '/Auswertung/checkpoint_model_1'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Fit the model
history_1 = model_1.fit(train_data_augmented,
                        epochs=8,
                        steps_per_epoch=len(train_data_augmented),
                        validation_data=validation_data_augmented,
                        validation_steps=len(validation_data_augmented),
                        callbacks=[reduce_lr, earlyStopping])

#############################################################
# Transfer learning 'Create model'
#############################################################

def create_model(model_url, num_classes=10, input_shape=(224, 224)):
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

#############################################################
# Transfer learning
#############################################################

# Resnet 50 V2 feature vector
resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"

# EfficientNet0 feature vector
efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"

# Create model
model2 = create_model(resnet_url, 2)
model2.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

# Fit model
history2 = model2.fit(train_data_augmented,
                      epochs=4,
                      steps_per_epoch=len(train_data_augmented),
                      validation_data=validation_data_augmented,
                      validation_steps=len(validation_data_augmented))

#############################################################
# Transfer learning - Feature extraction
#############################################################

# Build data augmentation layer
data_augmentation = Sequential([
  preprocessing.RandomFlip('horizontal'),
  preprocessing.RandomHeight(0.2),
  preprocessing.RandomWidth(0.2),
  preprocessing.RandomZoom(0.2),
  preprocessing.RandomRotation(0.2),
  preprocessing.Rescaling(1./255)
], name="data_augmentation")

# 1. Create base model with tf.keras.applications
base_model = tf.keras.applications.resnet50.ResNet50(include_top=False)

# 2. Freeze the base model (so the pre-learned patterns remain)
base_model.trainable = True

for layer in base_model.layers[:10]:
    layer.trainable = False

# 3. Create inputs into the base model
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
x = data_augmentation(inputs)

# 5. Pass the inputs to the base_model (note: using tf.keras.applications, EfficientNet inputs don't have to be normalized)
x = base_model(inputs)
# Check data shape after passing it to base_model
print(f"Shape after base_model: {x.shape}")

# 6. Average pool the outputs of the base model (aggregate all the most important information, reduce number of computations)
x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
print(f"After GlobalAveragePooling2D(): {x.shape}")

# 7. Create the output activation layer
outputs = tf.keras.layers.Dense(2, activation="sigmoid", name="output_layer")(x)

# 8. Combine the inputs with the outputs into a model
model3 = tf.keras.Model(inputs, outputs)

# 9. Compile the model
model3.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# 10. Fit the model (we use less steps for validation so it's faster)
history3 = model3.fit(train_data,
                                 epochs=5,
                                 steps_per_epoch=len(train_data),
                                 validation_data=val_data,
                                 # Go through less of the validation data so epochs are faster (we want faster experiments!)
                                 validation_steps=len(val_data))
