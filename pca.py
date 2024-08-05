
#1. Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# DataSet
veriler = pd.read_csv('Wine.csv')
X = veriler.iloc[:, 0:13].values
y = veriler.iloc[:, 13].values

# Splitting Training and Test Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

# LR Before PCA Transformation
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

# LR After PCA Transformation
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)

# Predicts
y_pred = classifier.predict(X_test)

y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
# Actual / Result without PCA 
print('gercek / PCAsiz')
cm = confusion_matrix(y_test,y_pred)
print(cm)

# Actual / Result after PCA
print("gercek / pca ile")
cm2 = confusion_matrix(y_test,y_pred2)
print(cm2)

# After PCA / Before PCA
print('pcasiz ve pcali')
cm3 = confusion_matrix(y_pred,y_pred2)
print(cm3)

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)

X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# LR After LDA Transformation
classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda,y_train)

# LDA Predict
y_pred_lda = classifier_lda.predict(X_test_lda)


# After LDA / Before LDA
print('ldasiz ve ldali')
cm4 = confusion_matrix(y_pred_lda,y_pred)
print(cm4)






























