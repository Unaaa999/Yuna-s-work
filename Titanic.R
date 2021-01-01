
#1. Read in training dataset, understand its distribution, columns, and missing value 


setwd("E:\\Kaggle\\1. Titanic")
train<-read.csv("train.csv", header=TRUE)
# always a good practice to cast all "" to NA
train[train==""] <- NA


#nrow(train)
#ncol(train)
dim(train)

summary(train)
library(naniar)
miss_var_summary(train)

#sum(train$Cabin == " ")
# sum(train$Ticket == " ")
# sum(train$Embarked == " ")
# sum(train$Name == " ")
# sum(train$Sex == " ")

# Tip 1: How to replace the value
# train$Cabin[train$Cabin==""] = NA


#1. Embarked 2 is missing ->low volume, we can get rid of this 
train2<-subset(train, Embarked!="")

#2. Age needs to be cleaned 177/891, 20% missing -> Is it important? yes. Therefore we need to do sth about it
# Solution 1: replace with mean or median
# one would be more accurate in imputing missing ages by calculating mean age of a group of 'similar passengers
# Solution 2: Random value 
# Solution 3: imputation
# Method:https://www.kaggle.com/parulpandey/a-guide-to-handling-missing-values-in-python

# First take a look at the correlation between the predictors 
# transfer the categorical into numeric
train2$Sex[train2$Sex=="male"] = 0
train2$Sex[train2$Sex=="female"] = 1

unique(train2$Sex)
unique(train2$Embarked)
train2$Embarked[train2$Embarked=="C"] = 0
train2$Embarked[train2$Embarked=="Q"] = 1
train2$Embarked[train2$Embarked=="S"] = 2

train2$Sex<-as.numeric(train2$Sex) 
train2$Embarked<-as.numeric(train2$Embarked)

# C = Cherbourg, Q = Queenstown, S = Southampton
library(dplyr)
train3<-subset(train2, !is.na(Age))
m<-train3 %>% select(3, 5:8,10,12)


head(round(m,2))
miss_var_summary(m)

#Correlation plot
library(corrplot)
#Remember to make correlation matrix before visilazing it
M<-cor(m)
corrplot(M, method="circle")

# Base on this plot, Age is correclated with pclass, meaning the better the class(lower the value), the older the age is
# Another observation is more Sibings the younger the age is
# Parch and Fare are also kinda correlated, Embarked is not so correlated with Age
# Final candidate predictor for Age: Pclass,SibSp, Parch

# Second, use these predictors to impute the missing value of Age
train2$Age_ind[is.na(train2$Age)] = 0
train2$Age_ind[!is.na(train2$Age)] = 1
lm1<-lm(Age ~ Pclass+SibSp+Parch, train3)
summary(lm1)
predict(lm1, newdata=train2)

lm2<-lm(Age ~ Pclass+SibSp+Parch+Sex+Fare, train3)
summary(lm2)
predict(lm2, newdata=train2)

lm3<-lm(Age ~ Pclass+SibSp+Sex, train3)
summary(lm3)
train2$new_Age<-predict(lm3, newdata=train2)


#Looks like none of the model is good, maybe we can introduce title instead
# combines some passenger titles
library(randomForest)
library(stringr)
train2$title <- str_sub(train2$Name, str_locate(train2$Name, ",")[ , 1] + 2, str_locate(train2$Name, "\\.")[ , 1] - 1)

male_noble_names <- c("Capt", "Col", "Don", "Dr", "Jonkheer", "Major", "Rev", "Sir")
train2$title[train2$title %in% male_noble_names] <- "male_noble"

female_noble_names <- c("Lady", "Mlle", "Mme", "Ms", "the Countess")
train2$title[train2$title %in% female_noble_names] <- "female_noble"




lm4<-lm(Age ~ Pclass+SibSp+Sex+title, train2)
summary(lm4)
#model improved!
train2$new_Age<-predict(lm4, newdata=train2)
# But there is some negative value, titleMiss,titleMr,titleMrs are not so significant, we can group them
# train2$title[train2$title=="Miss"|train2$title=="Mr"|train2$title=="Mrs"] <- "Others"
unique(train2$title)

lm5<-lm(Age ~ Pclass+SibSp+title, train2)
summary(lm5)
train2$new_Age<-predict(lm5, newdata=train2)
summary(train2$new_Age)
# Even worse, looks like we need to fit a non-zero distribution
ggplot(train2,stat="count") + geom_bar(aes(x = title))
train2$title[train2$title=="female_noble"|train2$title=="male_noble"] <- "noble"

ggplot(train2,stat="count") + geom_bar(aes(x = title))


ggplot(train, aes(x=Age)) + geom_histogram(binwidth=8)
library(moments)
skewness(train3$Age)  
# If the skewness is between -0.5 and 0.5, the data are fairly symmetrical. If the skewness is between -1 and – 0.5 or between 0.5 and 1, the data are moderately skewed.


# Fit a Gamma model first 
lm6 <- glm(formula = Age ~ Pclass+SibSp+title,
                         family  = Gamma(link = "log"),
                         data    = train2)
summary(lm6)
train2$new_Age<-predict(lm6, newdata=train2,type="response")
summary(train2$new_Age)
rsq(lm6,adj=TRUE)

ggplot(train2, aes(x=Age)) + geom_histogram(binwidth=8)
ggplot(train2, aes(x=new_Age)) + geom_histogram(binwidth=8)

summary(train3$Age)
summary(train2$new_Age)

# The model is not able to pick up the signal at lower and upper end, let's clean the data further

# young age
Test1<-subset(train2, train2$Age<=10)
ggplot(Test1,stat="count") + geom_bar(aes(x = title))
summary(Test2$Age)
Test2<-subset(train2, train2$title=="Master")
Test3<-subset(train2, train2$title=="Miss")
summary(Test3$Age)

# Younger age boy tends  to be called Master, whereas younger girl can be called Miss, but miss can age range from 0 to 63
# So i decided to create another group called 'young miss'
train2$titlenew[train2$Age<=12 & train2$title=="Miss"] <- "young kids"
unique(train2$titlenew)
train2$title[train2$Age<=12 & train2$title=="Miss"] <- "young kids"
unique(train2$title)
train2$title[train2$title=="Master"] <- "young kids"
unique(train2$title)
ggplot(train2,stat="count") + geom_bar(aes(x = title))


lm7 <- glm(formula = Age ~ Pclass+title+Fare,
           family  = Gamma(link = "log"),
           data    = train2)
summary(lm7)
train2$new_Age<-predict(lm7, newdata=train2,type="response")
summary(train2$new_Age)
rsq(lm7,adj=TRUE)

# Old age
Test1<-subset(train2, train2$Age>=60)
# Not much to say here
train2<-train2[1:15]


lm8 <- glm(formula = Age ~ Pclass+title+SibSp+Parch,
           family  = Gamma(link = "log"),
           data    = train2)
summary(lm8)
train2$new_Age<-predict(lm8, newdata=train2,type="response")

summary(train2$new_Age)
summary(train2$Age)
rsq(lm8,adj=TRUE)

train2$diff<-train2$new_Age-train2$Age
ggplot(train2, aes(x=diff)) + geom_histogram(binwidth=8)
# Tend to overstate...

ggplot(train2, aes(x=Age)) + geom_histogram(binwidth=8)
ggplot(train2, aes(x=new_Age)) + geom_histogram(binwidth=8)











library(Hmisc)
# impute<-subset(train2, is.na(Age))

# argImpute() automatically identifies the variable type and treats them accordingly.
# https://www.rdocumentation.org/packages/Hmisc/versions/4.3-0/topics/aregImpute


impute_arg <- aregImpute(~ Age ,
                            data = train2, n.impute = 5)



#Cabin, 687 is missing 







library(ggplot2)
ggplot(train, aes(x=Age)) + geom_histogram(binwidth=1)





#Correlation map
head(train)
library(corrplot)
M<-cor(train)
cormat<-round(cor(train),2)


