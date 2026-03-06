import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from feature_extraction import extract_feature

dataset = "dataset"

X = []
y = []

for root, dirs, files in os.walk(dataset):

    for file in files:

        if file.endswith(".wav"):

            emotion = file.split("-")[2]

            file_path = os.path.join(root, file)

            feature = extract_feature(file_path)

            X.append(feature)
            y.append(emotion)

X = np.array(X)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model = SVC(kernel="linear")

model.fit(x_train, y_train)

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

pickle.dump(model, open("emotion_model.pkl", "wb"))