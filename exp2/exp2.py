import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv("svm.csv")

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

model = SVC(kernel="linear")

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy * 100, "%")

new_data = []

print("\nEnter values for prediction:")

for column in x.columns:
    value = float(input(f"{column}: "))
    new_data.append(value)

prediction = model.predict([new_data])

print("Predicted Output:", prediction[0])
