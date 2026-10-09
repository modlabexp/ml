import os
import kagglehub
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

path = kagglehub.dataset_download("datamunge/sign-language-mnist")

train = pd.read_csv(os.path.join(path, "sign_mnist_train.csv"))
test = pd.read_csv(os.path.join(path, "sign_mnist_test.csv"))

x_train = train.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255
x_test = test.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255

y_train = to_categorical(train["label"], 25)
y_test = to_categorical(test["label"], 25)

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(25, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=5, verbose=0)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

print("Accuracy:", round(accuracy * 100, 2), "%")

pred = model.predict(x_test, verbose=0)

print(classification_report(
    np.argmax(y_test, axis=1),
    np.argmax(pred, axis=1)
))
