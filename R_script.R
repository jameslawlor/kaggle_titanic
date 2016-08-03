# 
library(randomForest)
library(caret)
setwd("~/Documents/Kaggle_Titanic_R")
train <- read.csv("~/Documents/Kaggle_Titanic_R/train.csv")
test <- read.csv("~/Documents/Kaggle_Titanic_R/test.csv",stringsAsFactors = T)
test$Survived <- NA
combi <- rbind(train,test)
combi$Name <- as.character(combi$Name) # convert factor to str
combi$Title <- sapply(combi$Name, FUN = function(x){strsplit(x, split='[,.]')[[1]][[2]]}) # create Title col via sapply
combi$Title <- sub(' ','',combi$Title)
combi$Title[combi$Title == 'Mlle'] <- 'Miss' # French titles
combi$Title[combi$Title == 'Mme'] <- 'Mrs'
combi$Title[combi$Title %in% c('Capt','Col','Don','Jonkheer','Major','Rev')] <- 'Sir' # combine rare Honorifics
combi$Title[combi$Title %in% c('Dona','the Countess')] <- 'Lady'
combi$Title <- factor(combi$Title) # convert back to factor
combi$FamilySize <- combi$SibSp + 1 + combi$Parch
combi$Surname <- sapply(combi$Name, FUN = function(x){strsplit(x, split='[,.]')[[1]][[1]]})
combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep="")
combi$FamilyID[combi$FamilySize <= 2] <- "Small"
famIDs <- data.frame(table(combi$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] <- 'Small' # assign buggy small families that didn't work on above filter
combi$FamilyID <- factor(combi$FamilyID)
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                data=combi[!is.na(combi$Age),], 
                method="anova")
combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),]) # run predictions on ages
which(combi$Embarked == '') # assign the two missing values in "Embarked" col to most common "S" value
combi$Embarked[c(62,830)] = "S"
combi$Embarked <- factor(combi$Embarked)
# fill in single missing Fare value with median fare
which(is.na(combi$Fare)) # returns 1044
combi$Fare[1044] <- median(combi$Fare,na.rm=T)
# random forest algo can only handle 32 factors max, our family feature has nearly 61! so increase "Small" value to size 3
combi$FamilyID2 <- combi$FamilyID # copy
combi$FamilyID2 <- as.character(combi$FamilyID2)
combi$FamilyID2[combi$FamilySize <= 3] <- "Small"
combi$FamilyID2 <- factor(combi$FamilyID2)
str(combi$FamilyID2) # check , we have only 22 levels of this factor now
combi$Deck <- as.factor(sapply(combi$Cabin, FUN = function(x){ substr(x,1,1) }))
# now run RF
set.seed(0)
train <- combi[1:891,]
test <- combi[892:1309,]
#### using party library (like RF but uses stats instead of a 'purity test' to split the DTrees)
library(party)
library(caret)
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare +
                 Embarked + Title + FamilySize + FamilyID + Deck,
               data = train, 
               controls=cforest_unbiased(ntree=1500, mtry=3,trace=T))
Prediction <- predict(fit, test, OOB=TRUE, type = "response")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = 'adaboost', row.names= F)
