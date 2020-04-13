from __future__ import absolute_import, division, print_function, unicode_literals

import time

import PIL.Image as Image
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

# Download the classifier
classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"  # @param {type:"string"}
IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE + (3,))
])

print(classifier.summary())

# Run in single image
# Download image to try the model.
grace_hopper = tf.keras.utils.get_file(
    'image.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
plt.figure(1)
plt.imshow(grace_hopper)
print('grace_hopper shape: {}'.format(type(grace_hopper)))

grace_hopper = np.array(grace_hopper) / 255.0
print(grace_hopper.shape)

result = classifier.predict(grace_hopper[np.newaxis, ...])
print('Prediction: \nShape: {}\n{}'.format(result.shape, result))

predicted_class = np.argmax(result[0], axis=-1)
print('predicted class: {}'.format(predicted_class))

# Decode the prediction
labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

plt.figure(2)
plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
plt.title("Prediction: " + predicted_class_name.title())
plt.show()

##############################################
# Simple transfer learning
##############################################
data_root = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)

# The result is a iterator that return pairs (image, label)
for image_batch, label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break

aux = 0
for i, (image_batch, label_batch) in enumerate(image_data):
    plt.subplot(2, 2, (i + 1))
    plt.imshow(image_batch[0])
    plt.xlabel(label_batch[0])
    if i == 3:
        break
plt.show()

result_batch = classifier.predict(image_batch)
print(result_batch.shape)

predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
print(predicted_class_names)

plt.figure(figsize=(10, 9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6, 5, n + 1)
    plt.imshow(image_batch[n])
    plt.title(predicted_class_names[n])
    plt.axis('off')
    plt.suptitle("ImageNet predictions")
plt.show()

##############################################
# Download the headless model
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"  # @param {type:"string"}

# Create the feature extractor.
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224, 224, 3))
feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)

# Freeze the variables in the feature extractor layer,
# so that the training only modifies the new classifier layer.
feature_extractor_layer.trainable = False

# Wrap the hub layer in a tf.keras.Sequential model,
# and add a new classification layer.
model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(image_data.num_classes)
])

print(model.summary())

flower_predictions = model(image_batch)
print(flower_predictions.shape)

# Train model
# Use compile to configure the training process:
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['acc']
)


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()


steps_per_epoch = np.ceil(image_data.samples / image_data.batch_size)

batch_stats_callback = CollectBatchStats()

history = model.fit_generator(image_data, epochs=2,
                              steps_per_epoch=steps_per_epoch,
                              callbacks=[batch_stats_callback])

# Show that the model is making progress on the task.
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0, 2])
plt.plot(batch_stats_callback.batch_losses)

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0, 1])
plt.plot(batch_stats_callback.batch_acc)

plt.show()

# Check prediction
class_names = sorted(image_data.class_indices.items(), key=lambda pair: pair[1])
class_names = np.array([key.title() for key, value in class_names])
print(class_names)

predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]
label_id = np.argmax(label_batch, axis=-1)

# Plot result
plt.figure(figsize=(10, 9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6, 5, n + 1)
    plt.imshow(image_batch[n])
    color = "green" if predicted_id[n] == label_id[n] else "red"
    plt.title(predicted_label_batch[n].title(), color=color)
    plt.axis('off')
    plt.suptitle("Model predictions (green: correct, red: incorrect)")
plt.show()

# Export model
t = time.time()

export_path = "./mymodel_1"
model.save(export_path)
print(export_path)

# Load model
reloaded = tf.keras.models.load_model(export_path)
result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)
print(abs(reloaded_result_batch - result_batch).max())
