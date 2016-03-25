"""
=================================================
Kaggle Competition - Titanic Dataset
=================================================
"""
# Author: James Lawlor <jalawlor@tcd.ie>

#######################################################################
#       Import and tidy the data
######################################################################

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd


def create_submission(clf, train, test, predictors, f="kaggle.csv"):
    """
    creates Kaggle submission file
    """
    clf.fit(train[predictors], train["Survived"])
    predictions = clf.predict(test[predictors].astype(float))
    submission = pd.DataFrame({
                    "PassengerId": test["PassengerId"], 
                    "Survived": predictions    })
    
    submission.to_csv(f, index=False)

    return

def process(db,fare_med=0,age_med=0):
    """
    preprocess data set
    """
    # fill NAN ages with median - we should make this better at some point
    if fare_med:
        db["Fare"] = db["Fare"].fillna(fare_med)
    else:
        db["Fare"] = db["Fare"].fillna(db["Fare"].median())

    if age_med:
        db["Age"] = db["Age"].fillna(age_med)
    else:
        db["Age"] = db["Age"].fillna(db["Age"].median())
    # Replace all the occurences of male with the number 0.
    db.loc[db["Sex"] == "male", "Sex"] = 0 
    db.loc[db["Sex"] == "female", "Sex"] = 1
    
    # tidy embarked bit
    db["Embarked"] = db["Embarked"].fillna("S") # improve later
    db.loc[db["Embarked"] == "S", "Embarked"] = 0
    db.loc[db["Embarked"] == "C", "Embarked"] = 1
    db.loc[db["Embarked"] == "Q", "Embarked"] = 2

    return db

train = pd.read_csv("train.csv")
test = pd.read_csv('test.csv')
#fare_med_temp = train["Fare"].median()
#age_med_temp = train["Age"].median()

train = process(train)
test = process(test)
#test = process(test,fare_med_temp,age_med_temp)


predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
clf = GradientBoostingClassifier(random_state=1, n_estimators=15, max_depth=6)
clf = RandomForestClassifier(
#        random_state=3,
        n_estimators=300,
        min_samples_split=4,
        min_samples_leaf=2
    )

#clf = SVC(C=100.0,gamma=0.001,kernel='rbf')
#clf = SVC(C=1.0,kernel='linear')


scores = cross_validation.cross_val_score(
        clf,
        train[predictors],
        train["Survived"],
        cv=3
    )

print(scores.mean())
#clf = SVC(C=100.0,gamma=0.001,kernel='rbf')

create_submission(clf, train, test, predictors)
