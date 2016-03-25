"""
=================================================
Kaggle Competition - Titanic Dataset
=================================================
"""
# Author: James Lawlor <jalawlor@tcd.ie>

#######################################################################
#       Import and tidy the data
######################################################################
import pandas as pd
import numpy as np

titanic = pd.read_csv("train.csv")
#print titanic.describe()
 # fill NAN ages with median - we should bootstrap this later
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
# Replace all the occurences of male with the number 0.
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0 
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# tidy embarked bit
titanic["Embarked"] = titanic["Embarked"].fillna("S") # bootstrap later
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

################################################################
#       Machine Learning bit
################################################################


# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold

# The columns we'll use to predict the target, they correspond to titanic column names
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
kf = KFold(titanic.shape[0], n_folds=3, random_state=1) 
# shape[0] = how many rows
# random_state = integer is the seed, by setting equal to 1 we get same splits every time we run

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    #gives probability of survival:
    predictions.append(test_predictions)

# The predictions are in three separate numpy arrays.  Concatenate them into one.  
# We concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)

from sklearn import cross_validation
from sklearn import linear_model
# Initialize our algorithm
alg = linear_model.LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

#####################################################################################
#####################################################################################
#               TEST THE TEST DATA AND EXPORT TO CSV
#####################################################################################


titanic_test = pd.read_csv("test.csv")
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0 
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2
# Initialize the algorithm class
alg = linear_model.LogisticRegression(random_state=1)

# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set.
predictions = alg.predict(titanic_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
            "PassengerId": titanic_test["PassengerId"],
                    "Survived": predictions
                        })
submission.to_csv("kaggle.csv", index=False)
