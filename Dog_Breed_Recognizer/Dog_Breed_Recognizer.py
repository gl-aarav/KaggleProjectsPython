#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from keras.preprocessing import image
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
#%% md
# Importing required libraries
#%%
# Read the labels.csv file and checking shape and records
labels_all = pd.read_csv('labels.csv')
print(labels_all.shape)
labels_all.head()
#%% md
# Loading the labels data into dataframe and viewing it . Here we analyze that labels contain 10222 rows and 2 columns
#%%
# Visualize the number of each breed
breeds_all = labels_all['breed']
breeds_counts = breeds_all.value_counts()
breeds_counts.head()
#%% md
# Finding out the count per class i.e. total data in each class using the value_counts() function
#%%
# Selecting first 3 breeds
CLASS_NAMES = ['scottish_deerhound', 'maltest_dog', 'bernese_mountain_dog']
labels = labels_all[(labels_all['breed'].isin(CLASS_NAMES))]
labels = labels.reset_index()
labels.head()
#%%
# Creating numpy matrix with zeros
X_data = np.zeros ((len(labels), 224, 224, 3), dtype='float32')
# One hot encoding
Y_data = label_binarize(labels[ 'breed'], classes = CLASS_NAMES)

# Reading and converting image to numpy array and normalizing dataset
for i in tqdm(range(len(labels))):
    img = image. load_img( 'train/%s.jpg' % labels['id'][i], target_size=(224, 224))
    img = image. img_to_array(img)
    x = np.expand_dims(img.copy(), axis=0)
    X_data [i] = x / 255.0
# Printing train image and one hot encode shape & size
print('\nTrain Images shape: ',X_data.shape,' size: {:,}'. format(X_data.size))
print('One-hot encoded output shape: ',Y_data.shape,' size: {:,}'.format(Y_data.size))
#%%
# Building the Model
model = Sequential()

model.add(Conv2D(filters=64, kernel_size = (5,5), activation='relu', input_shape=(224,224,3)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size = (3,3), activation='relu', kernel_regularizer='l2'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=16, kernel_size = (7,7), activation='relu', kernel_regularizer='l2'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=8, kernel_size = (5,5), activation='relu', kernel_regularizer='l2'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
model.add(Dense(64, activation='relu', kernel_regularizer='l2'))
model.add(Dense(len(CLASS_NAMES), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001),  metrics=['accuracy'])

model.summary()
#%%
# Splitting the data into train and testing set
X_train_and_val, X_test, Y_train_and_val, Y_test = train_test_split(X_data, Y_data, test_size=0.1)
# Splitting data into train and validation set
X_train, X_val, Y_train, Y_val = train_test_split(X_train_and_val, Y_train_and_val, test_size=0.2)
#%%
# Training the model
history = model.fit(X_train, Y_train, batch_size=128, epochs=100, validation_data=(X_val, Y_val))
#%%
# Plot the training history
plt.figure(figsize=(12,5))
plt.plot(history.history['accuracy'], color = 'red')
plt.plot(history.history['val_accuracy'], color = 'blue')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'])
plt.show()
#%%
