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
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

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

