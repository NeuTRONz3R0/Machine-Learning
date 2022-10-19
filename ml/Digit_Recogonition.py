import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, MaxPool2D,  Conv2D, Dropout, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape, y_train.shape, x_test.shape, y_test.shape

#just to visualise the dataset
def input_plotter(i):
  plt.imshow(x_train[i], cmap = "binary")
  plt.title(y_train[i])
  plt.show()
for i in range(10):
  input_plotter(i)
  
#preprocessing the data
x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255
#reshape images to (28,28,1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

#model Creation
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (28,28,1), activation = 'relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ['accuracy'])


from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'validation_acc', min_delta = 0.01, patience = 4, verbose = 1)
mc = ModelCheckpoint("bestmode.h5", monitor = 'validation_acc', verbose = 1, save_best = True)
cb = [es, mc]


history = model.fit(x_train, y_train, epochs= 50, validation_split= 0.3)
score = model.evaluate(x_test, y_test)
print("model accuracy is", score[1])
model.save("bestmodel.h5")


#test model by inputting a single image 
#NOTE: image should be resized(28,28,1)
import cv2
import numpy as np
from PIL import Image
from keras.saving.save import load_model
image = Image.open("/content/download (1).jpg")
print(image.format)
print(image.size)
image = np.array(image)
#image = image[:,:,0]
#image = np.pad(image, (10,10), 'constant',constant_values=0)
#image = cv2.imread("/content/download (1).jpg") 
#image = cv2.resize(image, (28,28))
#image = cv2.resize(image, (28,28))/225
#image = np.reshape(image,(1,28,28,1))
image = cv2.imread("image DIR", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28))
image = image.astype('float32')
image = image.reshape(1, 28, 28, 1)
image = 255-image
image /= 255
model = load_model('/content/bestmodel.h5')
pred = model.predict(image.reshape(1, 28, 28, 1), batch_size=1)

#pred_class = list(pred).index(max(pred))
print(pred.argmax())

#submitted By Ujjwal Deep
