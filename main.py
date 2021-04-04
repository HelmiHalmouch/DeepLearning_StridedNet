'''
Application of StridedNet for costum image classification 
Here is the main code which apply the training neural network step 

GHANMI Helmi 
02/01/2019

'''

import cv2
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.regularizers import l2
from imutils import paths
from stridednet import StridedNet

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-e", "--epochs", type=int, default=50,
                help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="result_train.png",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

#-------------------Preprocessing and split of datasets---------------#
# initialize the set of labels from the CALTECH-101 dataset we are
# going to train our network on
LABELS = set(["Faces", "Leopards", "Motorbikes", "airplanes"])
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
# print(imagePaths)
data = []  # A list to hold our images that our network will be trained on
labels = []  # A list to hold our class labels that correspond to the data

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # if the label of the current image is not part of the labels
    # are interested in, then ignore the image
    if label not in LABELS:
        continue
    # load the image and resize it to be a fixed 96x96 pixels,
    # ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (96, 96))
    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print('The classe are :', lb.classes_)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

#-------------------------Compile and train the CNN striednet model -------------#
# initialize the optimizer and model
print("[INFO] compiling model...")
# optimizer
opt = Adam(lr=1e-4, decay=1e-4 / args["epochs"])
# build the model
model = StridedNet.build(width=96, height=96, depth=3,
                         classes=len(lb.classes_), reg=l2(0.0005))

# compile the model
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

# train the network model
print("[INFO] training network for {} epochs...".format(args["epochs"]))

model_fit_generator = model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), steps_per_epoch=len(trainX) // 32,
                                          epochs=args["epochs"])

# evaluate the model
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

# Save the model h5
model.save('trained_custom_model.h5')
model.save_weights("trained_custom_weight_model.h5")
print("Saved model .h5 to disk")

# save the mode to json file
model_json = model.to_json()
with open("trained_jsonmodel.json", "w") as json_file:
    json_file.write(model_json)
print("Saved model json to disk")

# plot the training loss and accuracy
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), model_fit_generator.history[
         "loss"], label="train_loss")
plt.plot(np.arange(0, N), model_fit_generator.history[
         "val_loss"], label="val_loss")
plt.plot(np.arange(0, N), model_fit_generator.history[
         "acc"], label="train_acc")
plt.plot(np.arange(0, N), model_fit_generator.history[
         "val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('result_of_training.png')
