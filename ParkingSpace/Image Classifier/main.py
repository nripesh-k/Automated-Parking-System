import os
import pickle

#skimage
from skimage.io import imread
from skimage.transform import resize

#numpy
import numpy as np

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# preparing the data
inp_dir = './all-data'
cats = ['empty', 'not_empty']

data = []
labels = []

for idx, category in enumerate(cats):
    for file in os.listdir(os.path.join(inp_dir, category)):
        img_path = os.path.join(inp_dir, category, file)

        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(idx)

data = np.asarray(data)
labels = np.asarray(labels)

# train / test splitting
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)

# training the classifier
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# testing its performance with test set
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('Accuracy: {}%'.format(str(score * 100)))

#saving the model
pickle.dump(best_estimator, open('./saved_model.p', 'wb'))
