import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = load_digits()

x = data.data / 16.0
y = data.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

model = Sequential([
    Dense(32, activation="relu", input_shape=(64,)),
    Dense(16, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x_train,
    y_train,
    epochs=20,
    verbose=0
)

loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("Accuracy:", round(accuracy * 100, 2), "%")

pred = model.predict(x_test[:1], verbose=0)

print("Predicted Digit:", np.argmax(pred))
