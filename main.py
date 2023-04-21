"""a tool that trains several machine learning models to perform the
task of classifying Kickstarter posts"""

import csv, nltk, sklearn.feature_extraction.text, sklearn.metrics, sklearn.model_selection, sklearn.neighbors, sklearn.neural_network, sklearn.tree, matplotlib.pyplot as plt, joblib
#nltk.download("punkt", quiet = True)
#nltk.download("stopwords", quiet = True)

from nltk.corpus import stopwords

try: 
  # List for descriptions
  text = []

  #list for status (dependent variable)
  y = []

  with open("kickstarter_data_full.csv", "r", encoding = "ISO-8859-1") as file:
    reader = csv.reader(file)
    for row in reader:
      desc = row[5]
      req = row[4]

      # Convert statuses to binary
      status = row[3]
      if status == "Funding Successful":
        status = 1
      else:
        status = 0

      text.append(desc)
      y.append(status)

    #create vectorizer to remove stopwords and create matrix
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words = stopwords.words("english"))
    vectors = vectorizer.fit_transform(text)
    x = vectors.toarray()

    # Split data into training and test portions
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y)

    # Decision tree
    dt_clf = sklearn.tree.DecisionTreeClassifier()
    dt_clf = dt_clf.fit(x_train, y_train)
    dt_predictions = dt_clf.predict(x_test)
    dt_accuracy = sklearn.metrics.accuracy_score(y_test, dt_predictions)
    print("DT accuracy:", dt_accuracy)

    # KNN
    knn_clf = sklearn.neighbors.KNeighborsClassifier(5)
    knn_clf = knn_clf.fit(x_train, y_train)
    knn_predictions = knn_clf.predict(x_test)
    knn_accuracy = sklearn.metrics.accuracy_score(y_test, knn_predictions)
    print("KNN accuracy:", knn_accuracy)

    # NN
    nn_clf = sklearn.neural_network.MLPClassifier()
    nn_clf = nn_clf.fit(x_train, y_train)
    nn_predictions = nn_clf.predict(x_test)
    nn_accuracy = sklearn.metrics.accuracy_score(y_test, nn_predictions)
    print("NN accuracy:", nn_accuracy)

    # If Else blocks to find most accurate model
    if dt_accuracy > knn_accuracy and dt_accuracy > nn_accuracy:
      predictions = dt_predictions
      clf = dt_clf
    elif knn_accuracy > dt_accuracy and knn_accuracy > nn_accuracy:
      predictions = knn_predictions
      clf = knn_clf
    else:
      predictions = nn_predictions
      clf = nn_clf

    # Show classification report for most accurate model
    print(sklearn.metrics.classification_report(y_test, predictions))

    # Show confusion matrix for most accurate model
    cm = sklearn.metrics.confusion_matrix(y_test, predictions)
    disp = sklearn.metrics.ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()  

    # Save joblib 
    joblib.dump(clf, "assignment5model.joblib")

except:
  print("could not connect to csv file")

