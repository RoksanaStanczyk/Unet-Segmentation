import glob

import callbacks as callbacks
import cv2
import numpy as np
import random
import numpy as np
import model
import tensorflow as tf
import os
import augmetation
from matplotlib import pyplot as plt
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split


# path = r"D:\Segmentation_project\nucleus_images_blue"
#
# def load_data(path):
#     images = sorted(glob(os.path.join(path,"images/*")))
#     masks = sorted(glob(os.path.join(path,"masks/*")))
#     return images, masks
#
# images, masks = load_data(path)
#
# def create_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)
#
# create_dir("new_data/images")
# create_dir("new_data/masks")
#
# ag = augmetation.augment_data(images, masks, "new_data", augment=True)
ignore_level = 0.003
def read_images(directory):
    for img in glob.glob(directory+"/*"):
        image = cv2.imread(img)
        gray = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2GRAY)
        resized_img = cv2.resize(gray / 255.0, (256, 256))

        yield resized_img
# read images and resize
images_list =  np.array(list(read_images("new_data/images")))
masks_list = np.array(list(read_images("new_data/masks")))

# normalization
images_list_gray = np.expand_dims(normalize(np.array(images_list), axis=1),3)
masks_list_gray = np.expand_dims((np.array(masks_list)),3) /255.
#
#split data
X_train, X_test, y_train, y_test = train_test_split(images_list_gray, masks_list_gray, test_size = 0.10, random_state = 0)

# show random image with mask
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.show()

IMG_HEIGHT = images_list_gray.shape[1]
IMG_WIDTH  = images_list_gray.shape[2]
IMG_CHANNELS = images_list_gray.shape[3]

def get_model():
    return model.simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()


callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'), # zatrzymuje w momencie kiedy nie ma progresu w nauce
        tf.keras.callbacks.TensorBoard(log_dir='logs')] # metryki i statystyki
#
# history = model.fit(X_train, y_train,
#                     batch_size = 16,
#                     verbose=1,
#                     epochs=50,
#                     validation_data=(X_test, y_test),
#                     shuffle=False)
#                     callbacks=callbacks)

# np.save('my_history.npy',history.history)
# np.save('my_history_50+aug2.npy',history.history)
# model.save('unet50+aug2.h5')
model = load_model('unet50+aug2.h5')
#
history=np.load('my_history_50+aug2.npy',allow_pickle='TRUE').item()
#Evaluate the model
#
#
    # ocena modelu
_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")
plt.figure()
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# # #
# # #
plt.figure()
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# # #
# # #
# # # IOU
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > ignore_level
#
intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU score is: ", iou_score)
#
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > ignore_level).astype(np.uint8)


test_img_other = cv2.imread('D:\Segmentation_project/zdjecie_testowe2.png', 0)
test_img_other = cv2.resize(test_img_other/255.0  , (256, 256))
test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other), axis=1),2)
test_img_other_input=np.expand_dims(test_img_other_norm, 0)
#
prediction_other = (model.predict(test_img_other_input)[0,:,:,0] > ignore_level).astype(np.uint8)
#
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('External Image')
plt.imshow(test_img_other)
plt.subplot(122)
plt.title('Prediction of external Image')
plt.imshow(prediction_other)
plt.show()
#
#
plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')
plt.subplot(234)
plt.title('External Image')
plt.imshow(test_img_other, cmap='gray')
plt.subplot(235)
plt.title('Prediction of external Image')
plt.imshow(prediction_other, cmap='gray')
plt.show()