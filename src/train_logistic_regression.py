import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import os
from sklearn.linear_model import LogisticRegression
import time
import psutil

with open(os.path.join("preprocessing", 'reviews.pickle'), 'rb') as handle:
    reviews = pickle.load(handle)

with open(os.path.join("preprocessing", 'labels.pickle'), 'rb') as handle:
    labels = pickle.load(handle)

confusion_matrix = {5: {5: 0, 4: 0, 3: 0, 2: 0, 1: 0},
                    4: {5: 0, 4: 0, 3: 0, 2: 0, 1: 0},
                    3: {5: 0, 4: 0, 3: 0, 2: 0, 1: 0},
                    2: {5: 0, 4: 0, 3: 0, 2: 0, 1: 0},
                    1: {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}}

labels = [int(label) for label in labels]

training_reviews = reviews[:350000]
training_labels = labels[:350000]

test_reviews = reviews[350000:]
gold_standard_labels = labels[350000:]

start_time = time.time()

# create count vectorizer and transform text data
vectorizer = CountVectorizer()
pred_X = vectorizer.fit_transform(training_reviews)

# train logistic regression classifier
clf = LogisticRegression(random_state=13, max_iter=1000)
clf.fit(pred_X, training_labels)

# transform test data using the same count vectorizer
test_X = vectorizer.transform(test_reviews)

# make predictions on test data
predictions = clf.predict(test_X)

predictions = predictions.tolist()


for i in range(len(predictions)):
    predicted_score = predictions[i]
    gold_standard_score = gold_standard_labels[i]
    confusion_matrix[gold_standard_score][predicted_score] += 1


# Confusion matrix:
# Rows = correct score (gold standard)
# Columns = predicted score
print("Confusion matrix:")

score_string = 5
for row in confusion_matrix.values():
    print(f"{str(score_string)}: {str(row)}")
    score_string -= 1
print("Rows = gold standard")
print("Columns = prediction")

print("")
print(
    f"Accuracy: {str(metrics.accuracy_score(gold_standard_labels, predictions))}"
)

end_time = time.time()
print(f"Execution time: {end_time - start_time}s")
print(
    f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MiB")
