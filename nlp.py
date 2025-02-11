# Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk # --> This lib will help to remove the unnecessary words which are not required in our prediction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB # --> For Naive Bayes Model
from sklearn.ensemble import RandomForestClassifier # --> For Decision Tree Model
from sklearn.neighbors import KNeighborsClassifier # --> For K-NN model
from sklearn.linear_model import LogisticRegression # --> For Logistic Regression Model
from sklearn.tree import DecisionTreeClassifier # --> For Random Forest model
from sklearn.svm import SVC # --> For Support Vector Machine Model
from sklearn.metrics import confusion_matrix, accuracy_score
# Import the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)

# Clean the text
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = [] # --> Stores the cleaned data
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
'''
Use the below print statement to check you steming
haas been applied correctly or not
-> Use for debuging purpose only.
'''
# print(corpus)

# Create a Bag of Words model
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values
# print(len(X[0])) // Debugging purpose to get the max_word to use in CountVectorizer

# Split the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Commented classification methods of the classification
# models to test each classification models for the 
# NLP, where we can also test accuray fit of the classification
# methods for the perfect fit for this project

# Train the Naive Base model on the Training Set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

'''# Train the Decision Tree model on the Training set
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier,fit(X_train, y_train)'''

'''# Train the K-NN model on the Training set
classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
classifier.fit(X_train, y_train)'''

'''# Train the Logistic Regression model on the Training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)'''

'''# Train the Random Forest model on the Training set
classifier = RandomForestClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)'''

'''# Train the Support Vector model on the Training set
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)'''

# Predict the Test set result
y_pred = classifier.predict(X_test)
# --> Just uncomment it to visualize your prediction datas
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)),1 ))

# Make the confusion matrix and predict the accuracy score
cm = confusion_matrix(y_test, y_pred)
# Just uncomment it to visualize your confusion matrix
'''
True Negative   |  False Positive
----------------|-----------------
False Negative  |  True Positive
'''
# print(cm)
print(f"The accuracy of this model is = {accuracy_score(y_test, y_pred)}")

