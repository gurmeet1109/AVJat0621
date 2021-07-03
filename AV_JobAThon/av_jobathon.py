'''
This file was made by Gurmeet (gurmeet1109@gmail.com, gurmeetarora9927@gmail.com) in May 2021
as a part of Analytics Vidya - Job-A-Thon competition
Rights are provided only to AnalyticsVidya for purposes it suits suitable as a part of the competition
'''


# here we will import the libraries used for machine learning
from datetime import datetime

import matplotlib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O, data manipulation
import matplotlib.pyplot as plt
import seaborn as sns # used for plot interactive graph.
from pandas import set_option
plt.style.use('ggplot')

from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.linear_model import LogisticRegression, LinearRegression  # to apply the Logistic regression
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold # for cross validation
from sklearn.model_selection import GridSearchCV # for tuning parameter
from sklearn.model_selection import RandomizedSearchCV  # Randomized search on hyper parameters.
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model

import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# Setting Pandas display options for proper display without truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('mode.chained_assignment', None)

timeprogramstart = datetime.now()

# Reading input files
train = pd.read_csv('./inputs/train_s3TEQDk.csv')
test = pd.read_csv('./inputs/test_mSzZ8RL.csv')
sample_submission = pd.read_csv('./inputs/sample_submission_eyYijxG.csv')

print("\nTrain File : ", train.shape)
print(train.sample(5))
print("\nTest file : ", test.shape)
print(test.sample(5))
print("\nSample Submission file :", sample_submission.shape)
print(sample_submission.sample(5))

# Ratio of null values
print("\nRatio of null values in train file : ", train.isnull().sum()/train.shape[0]*100)
print("\nRatio of null values in test file : ", test.isnull().sum()/test.shape[0]*100)

# ----------- Data Cleaning ---------
# There is only one column with Nan data

# Substituting Credit_Product with 'Mode" (Most Repeated Value)
train['Credit_Product'] = train['Credit_Product'].fillna(train['Credit_Product'].mode()[0])
test['Credit_Product'] = test['Credit_Product'].fillna(test['Credit_Product'].mode()[0])

print("\nNaN now in train : ", train['Credit_Product'].isnull().sum())
print("NaN now in test  : ", test['Credit_Product'].isnull().sum())

# ---- Feature Engineering  ------------
print("\nMaking Sure - No Anamolies")
print(train['Credit_Product'].value_counts())
print(test['Credit_Product'].value_counts())

# looking for other possible anamolies
print("\n", train['Gender'].value_counts())
print("\n", train['Occupation'].value_counts())
print("\n", train['Channel_Code'].value_counts())
print("\n", train['Is_Active'].value_counts())

print("\n", test['Gender'].value_counts())
print("\n", test['Occupation'].value_counts())
print("\n", test['Channel_Code'].value_counts())
print("\n", test['Is_Active'].value_counts())

print("\nSeems OK with other possible anamolies\n")


#   Exploratory Data Analysis

# Distribution plots
sns.countplot('Gender', data=train, palette='ocean')
plt.savefig("train_GenderMix.png")
plt.show()

sns.countplot('Occupation', data=train, palette='ocean')
plt.savefig("train_Occupation.png")
plt.show()

sns.countplot('Channel_Code', data=train, palette='ocean')
plt.savefig("train_ChannelCode.png")
plt.show()

sns.countplot('Is_Active', data=train, palette='ocean')
plt.savefig("train_ActiveStatus.png")
plt.show()

# Correlations matrix, defined via Pearson function
corr = train.corr()          # .corr is used to find correlation
f, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(corr, cbar = True,  square=True, annot=False, fmt='.1f',
            xticklabels= True, yticklabels=True, cmap="coolwarm", linewidths=.5, ax=ax)
plt.title('CORRELATION MATRIX - HEATMAP', size=18)
plt.savefig('Correlation_Matrix.png')
plt.show()

# One Hot Encoding
# OHE - Train Data
print("Before Hot Encoding")
print(train.head(5))
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

var_mod = ['Gender', 'Occupation', 'Channel_Code', 'Is_Active']
for i in var_mod:
    train[i] = le.fit_transform(train[i])

# One Hot Coding:
train = pd.get_dummies(train, columns=var_mod)
print("After Encoding - train Data")
print(train.head(5))

# OHE - test Data
for i in var_mod:
    test[i] = le.fit_transform(test[i])
test = pd.get_dummies(test, columns=var_mod)
print("After Encoding - test Data")
print(test.head(5))


#  ---------------- Building Model ----------------

# Removing all non-numeric columns
train = train.select_dtypes(exclude='object')
test = test.select_dtypes(exclude='object')

# Separate features and target
X = train.drop(columns=['Is_Lead'], axis=1)
y = train['Is_Lead']

# 20% data as validation set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=22)


# Model Building
features = X.columns
LR = LinearRegression(normalize=True)
LR.fit(X_train, y_train)
y_pred = LR.predict(X_valid)
coef = pd.Series(LR.coef_, features).sort_values()

# Barplot for coefficients
plt.figure(figsize=(8,5))
sns.barplot(LR.coef_,features)
plt.savefig("Coef_LinearRegression.png")


# RMSE - Root Mean Squared error
MSE= metrics.mean_squared_error(y_valid, y_pred)
from math import sqrt
rmse = sqrt(MSE)
print("Root Mean Squared Error:", rmse)

# Generate Submission file
submission = pd.read_csv('./inputs/sample_submission_eyYijxG.csv')
final_predictions = LR.predict(test)
submission['Is_Lead'] = final_predictions

#only positive predictions for the target variable
submission['Is_Lead'] = submission['Is_Lead'].apply(lambda x: 0 if x < 0 else x)
submission.to_csv('my_submission_LR.csv', index=False)
print(submission.head(10))



# Algorithm - Random Forest Classifier

# Create the random grid
param_dist = {'n_estimators': [50, 100, 150, 200, 250], "max_features": [1, 2, 3, 4, 5, 6, 7, 8, 9],
              'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9], "criterion": ["gini", "entropy"]}

rf = RandomForestClassifier()
rf_cv = RandomizedSearchCV(rf, param_distributions = param_dist, cv=5, random_state=0, n_jobs=-1)

rf_cv.fit(X, y)
print("Tuned Random Forest Parameters: %s" % (rf_cv.best_params_))

Ran = RandomForestClassifier(criterion='gini', max_depth=6, max_features=5, n_estimators=150, random_state=0)
Ran.fit(X_train, y_train)
y_pred = Ran.predict(X_valid)

#only positive predictions for the target variable
submission['Is_Lead'] = submission['Is_Lead'].apply(lambda x: 0 if x < 0 else x)
submission.to_csv('my_submission_RandomForest.csv', index=False)
print(submission.head(10))


submission['Is_Lead'] = y_pred
print('Accuracy:', metrics.accuracy_score(y_pred, y_valid))

## 5-fold cross-validation
cv_scores = cross_val_score(Ran, X, y, cv=5)

# Print the 5-fold cross-validation scores
print()
print(classification_report(y_valid, y_pred))
print()
print("Average 5-Fold CV Score: {}".format(round(np.mean(cv_scores),4)),
      ", Standard deviation: {}".format(round(np.std(cv_scores),4)))

plt.figure(figsize=(4, 3))
ConfMatrix = confusion_matrix(y_valid, Ran.predict(X_valid))
sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d",
            xticklabels = ['Non-default', 'Default'],
            yticklabels = ['Non-default', 'Default'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix - Random Forest")
plt.savefig('./plots/Conf_Mat_Random_Forest.png')
plt.show()


# ------------------------
print()
print()
print("Total Time Taken = ", (datetime.now() - timeprogramstart))
print()
print("----- End of Program ------")