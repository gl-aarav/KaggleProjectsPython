#%% md
# # Digit Recognizer
#%% md
# ## Import Libraries
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%% md
# ## Load the data
#%%
dataset_train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')
#%% md
# ### Data Exploration
#%%
X_train = dataset_train.iloc[:, 1:].values
y_train = dataset_train.iloc[:, 0].values
print(X_train[:25])
#%%
y_train = dataset_train.iloc[:, 0].values
y_train[:100]
#%% md
# # Using CNN
#%%
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.values.reshape(X_test.shape[0], 28, 28,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
#%%
from keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=10)
#%% md
# ## Build the model
#%%
# Simple CNN model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

#%%
# Evaluate the model
loss, accuracy = model.evaluate(X_train, y_train)
print(f'Training Accuracy: {accuracy*100:.2f}%')
#%% md
# ## Make Predictions
#%%
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
#%% md
# ## Prepare Submission
#%%
submission = pd.DataFrame({
    'ImageId': np.arange(1, len(predicted_classes) + 1),
    'Label': predicted_classes
})
submission.to_csv('submission.csv', index=False)
#%% md
# ## Visualize some predictions
#%%
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_test[i].reshape(28,28), cmap='gray')
    plt.title(f'Predicted: {predicted_classes[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()
#%% md
# 