install.packages("factoextra")
install.packages(("ggplot2"))
library(ggplot2)
library(factoextra)
library("FactoMineR")





heart <- read.csv("heart_failure_clinical_records_dataset.csv")
test <-read.csv("heart_failure_clinical_records_dataset.csv")

str(test)

head(heart)
colnames(heart)
str(heart)

# 299 rows, 13 variables. 'age', 'platelets' and 'serum_creatinine' are numeric whilst the rest are integers.
# 'sex', 'smoking', 'high_blood_pressure', 'diabetes', 'aneamia' & 'DEATH_EVENT' are all binary
# can regard binary data as categorical data

# NOTE: Death emvent is if the patient died within the follow up period
#       'time' is the follow up period in days

#therefore quantitative data is: age, creatinine_phosphokinase, ejection_fraction, platelets, serum_creatinine, serum_sodium

summary(heart)

table(is.na(heart))

sum(is.na(heart$age))
sum(is.na(heart$anaemia))
sum(is.na(heart$creatinine_phosphokinase))
sum(is.na(heart$diabetes))
sum(is.na(heart$ejection_fraction))
sum(is.na(heart$high_blood_pressure))
sum(is.na(heart$platelets))
sum(is.na(heart$serum_creatinine))
sum(is.na(heart$serum_sodium))
sum(is.na(heart$sex))
sum(is.na(heart$smoking))
sum(is.na(heart$time))
sum(is.na(heart$DEATH_EVENT))

#potentially no missing data??


table(heart$sex)

heart$anaemia <- as.factor(heart$anaemia)
heart$creatinine_phosphokinase <- as.numeric(heart$creatinine_phosphokinase)
heart$diabetes <- as.factor(heart$diabetes)
heart$ejection_fraction <- as.numeric(heart$ejection_fraction)
heart$high_blood_pressure <- as.factor(heart$high_blood_pressure)
heart$serum_sodium <- as.numeric(heart$serum_sodium)
heart$sex <- as.factor(heart$sex)
heart$smoking <- as.factor(heart$smoking)
heart$time <- as.numeric(heart$time)
heart$DEATH_EVENT <- as.numeric(heart$DEATH_EVENT)




#-----------------
# hist of ages, grouped by blood pressure


p <- ggplot(heart, aes(x = age)) + geom_histogram(aes(color = high_blood_pressure), 
                                             binwidth = 4, fill = "white") + ggtitle("Age & High Blood Pressure")

p + scale_fill_discrete(name = "Blood Pressure",
                        breaks = c("0", "1"),
                        labels = c("Non-high Blood Pressure", "High Blood Pressure"))


# age isn't completely symmetrical, considerably more people under hte age of 70
# perecentage of those with high blood pressure in an age group is higher in older age groups.
# key limitation is age starts at 40, therefore don't have data on younger people.

#hist of age and Death event:

ggplot(heart, aes(x = age, fill = smoking)) + geom_histogram(binwidth = 4) + ggtitle("Age & Smoking")

#across the ages, the 'total' amount of smokers in an age group doesn't massively fluctuate. 
# Percetange proportionality of the smokers across the age group looks to be consistent too


p <- ggplot(heart, aes(x = age, fill = sex)) + geom_histogram(binwidth = 4) + ggtitle("Variation of age by Sex") 

p + scale_fill_discrete(name = "Gender",
                        breaks = c("0", "1"),
                        labels = c("Female", "Male"))



ggplot(heart, aes(x = DEATH_EVENT, fill = c)) + geom_histogram(binwidth = 1, color = "Azure")

#as there are more males in the overall data, it can be reflected here as there are more males than females throughout the age groups
# especially throughout the older groups

table(heart$sex)

#--------
# Death event, with high blood pressure
# Death event, with smoking

ggplot(heart, aes(x= DEATH_EVENT)) + geom_bar(aes(color = high_blood_pressure), binwidth = 4, fill = "bisque") +
  ggtitle("Death Event & High Blood Pressure") + 
  xlab("0 = Survived, 1 = Died")


ggplot(heart, aes(x= DEATH_EVENT)) + geom_bar(aes(color = smoking), binwidth = 1, fill = "azure") + ggtitle("Death Event & Smoking") + xlab("0 = Survived, 1 = Died")

ggplot(heart, aes(x= DEATH_EVENT)) + geom_bar(aes(color = sex), binwidth = 4, fill = "darkgoldenrod1") + ggtitle("Death Event & Sex") + xlab("0 = Survived, 1 = Died")


# overall, the number of those who didn't die is just over double those who did die
# the % of those who had high blood pressure was higher in those who died, compared to those who didn't
# reviewing smokers, the ratio of smokers to non smokers, in those that died compared to those that lived is roughly equal.


#--------------
#differences by gender

ggplot(heart, aes(x= sex)) + geom_bar(aes(color = DEATH_EVENT), binwidth = 4, fill = "deeppink4") + ggtitle("Gender Distribution")

# considerably more males to females, almost 2:1 ratio




summary(heart$DEATH_EVENT)



str(heart)



#------ Factor Analysis ---------------------

pca <- prcomp(heart[,c(1,3,5,7:9,12)], scale = TRUE)

#choosing only the quantitative data

plot(pca$x[,1], pca$x[,3])


pca$rotation


fviz_eig(pca)    

#displaying the eigenvalues - PC1 has the biggest affect, with a value of just over 20%. 
#However, it's not 'clear-cut'. Other dimensions have a certain degree of variance too, with similar %'s.


fviz_pca_ind(pca,
             col.ind = "cos2",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE)

colnames(heart)



#with the package FactoMineR, we can use: PCA(X, scale.unit =TRUE, ncp =5, graph =TRUE)

FactoPca <- PCA(heart[,c(1,3,5,7:9,12)], scale.unit = TRUE, graph = FALSE)

print(FactoPca)

fviz_eig(FactoPca, addlabels = TRUE)
pca_eig_value <-get_eigenvalue(FactoPca)              #these two show the eigenvalues for each dimension!!
get_pca_var(FactoPca)
get_pca_ind(FactoPca)


fviz_pca_var(FactoPca)  
fviz_pca_ind(FactoPca)                # vizualising the individual results is unclean, stick with vizualising all variables

pca_eig_value
eig_value                             # to account for at least 80%, we need to use 5 principal components

var_pca <- get_pca_var(FactoPca)      #stored the variable information in a new object
var_pca$cor                           # from this, we can then vew the correlation for each variable, to the created PC's

var_pca$contrib                       # IMPORTANT: this shows the percentage contribution a variable has on each dimension!
                                                # in this instance, AGE seems consistently to have a large contribution of over 20%

                                      # However, there's also large contributions seen from ejection_fraction (46% )across dimension 2,
                                      # Also a large contribution from creatinine_phosphokinase (66%) in dimension 3
                                      # in summary, large contributions are seen from the variables ACROSS dimensions


library("corrplot")
corrplot(var_pca$cos2, is.corr = FALSE)   #this plot visualises the above plot^

# can use: var_pca(FactoPca$coord) to get the coordinates to create a scatter plot
# can be argued that variables correlating with the final dimensions can be dropped, therefore maybe drop platelets?


fviz_cos2(FactoPca, choice = 'var', axes = 1:4)       

#by changing the range of the axis above, we can see how many PC's are required for appropriate representation.
# even with 4 PC's, only Platelets and creatinine_phosphokinase have commendable representation. The rest fall below 60%.

fviz_pca_var(FactoPca, col.var = 'cos2',
             repel = TRUE)                    # further visualisation


#as we have 7 variables in question here, if they all have equal contribution then we'd expect (100/7 =) 14.28% from each.
# therefore, it's worthwhile seeing how each variable contributes compared to that figure:

fviz_contrib(FactoPca, choice = 'var', axes = 1, top = 14.28)     #visualising for PC1

fviz_contrib(FactoPca, choice = 'var', axes = 2, top = 14.28)     #visualising for PC2

fviz_contrib(FactoPca, choice = 'var', axes = 3, top = 14.28)     #visualising for PC3

#across both of these visualisation, platelets is drastically lower, therefore can drop.


ind_pca <- get_pca_ind(FactoPca)

#----------------------
# Attempt FAMD

str(heart)
famd <- FAMD(heart, ncp = 8, graph = FALSE)

# the default of 5 dimensions only account for 55% of the variance, increasing to 7 increases this to 70%
# had to change the binary variables to factors, and the integers to numeric.
heart$anaemia <- as.factor(heart$anaemia)
heart$creatinine_phosphokinase <- as.numeric(heart$creatinine_phosphokinase)
heart$diabetes <- as.factor(heart$diabetes)
heart$ejection_fraction <- as.numeric(heart$ejection_fraction)
heart$high_blood_pressure <- as.factor(heart$high_blood_pressure)
heart$serum_sodium <- as.numeric(heart$serum_sodium)
heart$sex <- as.factor(heart$sex)
heart$smoking <- as.factor(heart$smoking)
heart$time <- as.numeric(heart$time)
heart$DEATH_EVENT <- as.numeric(heart$DEATH_EVENT)


print(famd)

famd_eig <- get_eigenvalue(famd)
famd_var <- get_famd_var(famd)

famd_eig

# ----- visualisation ----

fviz_famd_var(famd)                   # unclear plot
fviz_screeplot(famd)
fviz_eig(famd) 


corrplot(famd_var$cos2, is.corr = FALSE) 

famd_var$contrib

fviz_contrib(famd, "var", axes = c(1,2,3))      #across the first three dimension, DE, Sex, Smoking & Time have higher than avg contributions to the dimensions

corrplot(famd_var$cos2, is.corr = FALSE)                  # in this instance, only time & Death event have notable contributions in D1
                                                #similarly, Sex and Smoking also have some contribution in D2. rest are negligible.

#famd suggests that the qualitative variables explain the the variance more notably than quantitative.



#--- drop platelets

heart_trimmed <- subset(heart, select = -(platelets))

heart <- heart_trimmed

colnames(heart)

# -- split into trainind and testing 

sample_size <- floor(0.8 * nrow(heart))

set.seed(111)
train_ind <- sample(seq_len(nrow(heart)), size = sample_size)

train <- heart[train_ind, ]
test <- heart[-train_ind, ]

nrow(train)
nrow(test)
#--- standardize the data

head(train)
# need to normalise: age, creatinine_phosphokinase, ejection_fraction, serum_creatinine, serum_sodium & time


train$age <- (train$age - min(train$age)) / (max(train$age) - min(train$age))
train$creatinine_phosphokinase <- (train$creatinine_phosphokinase - min(train$creatinine_phosphokinase)) / (max(train$creatinine_phosphokinase) - min(train$creatinine_phosphokinase))
train$ejection_fraction <- (train$ejection_fraction - min(train$ejection_fraction)) / (max(train$ejection_fraction) - min(train$ejection_fraction))
train$serum_creatinine <- (train$serum_creatinine - min(train$serum_creatinine)) / (max(train$serum_creatinine) - min(train$serum_creatinine))
train$serum_sodium <- (train$serum_sodium - min(train$serum_sodium)) / (max(train$serum_sodium) - min(train$serum_sodium))
train$time <- (train$time - min(train$time)) / (max(train$time) - min(train$time))


test$age <- (test$age - min(test$age)) / (max(test$age) - min(test$age))
test$creatinine_phosphokinase <- (test$creatinine_phosphokinase - min(test$creatinine_phosphokinase)) / 
  (max(test$creatinine_phosphokinase) - min(test$creatinine_phosphokinase))
test$ejection_fraction <- (test$ejection_fraction - min(test$ejection_fraction)) / 
  (max(test$ejection_fraction) - min(test$ejection_fraction))
test$serum_creatinine <- (test$serum_creatinine - min(test$serum_creatinine)) / 
  (max(test$serum_creatinine) - min(test$serum_creatinine))
test$serum_sodium <- (test$serum_sodium - min(test$serum_sodium)) / (max(test$serum_sodium) - min(test$serum_sodium))
test$time <- (test$time - min(test$time)) / (max(test$time) - min(test$time))


#---- create the NN model - 

library(neuralnet)

str(train)
str(heart)
str(test)

# convert Death Event to factor
train$DEATH_EVENT <- as.factor(train$DEATH_EVENT)

#change other factors (which are actually boolean/binary) into numerical 
# anaemia, diabetes, high_blood_pressure, sex, smoking

train$anaemia <- as.numeric(train$anaemia)
train$diabetes <- as.numeric(train$diabetes)
train$high_blood_pressure <- as.numeric(train$high_blood_pressure)
train$sex <- as.numeric(train$sex)
train$smoking <- as.numeric(train$smoking)

test$anaemia <- as.numeric(test$anaemia)
test$diabetes <- as.numeric(test$diabetes)
test$high_blood_pressure <- as.numeric(test$high_blood_pressure)
test$sex <- as.numeric(test$sex)
test$smoking <- as.numeric(test$smoking)



colnames(train)

model <- neuralnet(
  DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + diabetes + ejection_fraction 
  + high_blood_pressure + serum_creatinine + serum_sodium + sex + smoking + time,
  data = train, hidden = c(5, 5), threshold = 0.01,
  linear.output = FALSE
)


#                       notes: chosen 2 hidden layers of 5 each
#                              linear output is false as the desired output will be categorical as opposed to continuous (regarding death event)
str(train)


plot(model)

model$net.result
model$weights
model$result.matrix

# Test output prediction/build confusion matrix
head(train)

output <- compute(model, train[,-12])

p1 <- output$net.result
pred1 <- ifelse(p1 > 0.5, 1, 0)
tab1 <- table(pred1, train$DEATH_EVENT)                 
tab1                                           #confusion matrix of the current model on train data

Pred_accuracy1 <- sum(diag(tab1)) / sum(tab1)      #0.9539749
Pred_accuracy1

tab1 ; Pred_accuracy1

# on test:


test_output <- compute(model, test[,-12])

test_p1 <- test_output$net.result
test_pred1 <- ifelse(test_p1 > 0.5, 1, 0)
test_tab1 <- table(test_pred1, test$DEATH_EVENT)                 

test_tab1 ; test_Pred_accuracy1

test_Pred_accuracy1 <- sum(diag(test_tab1)) / sum(test_tab1)
test_Pred_accuracy1



#-------------------------------------

#now train and test prediction accuracy for variations in hidden layers
#model 2

model2 <- neuralnet(
  DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + diabetes + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium + sex + smoking + time,
  data = train, hidden = c(3), threshold = 0.01,
  linear.output = FALSE
)

plot(model2)

plot(model2)

M2output <- compute(model2, train[,-12])                         #first the train data
M2 <- M2output$net.result
M2_train_pred <- ifelse(M2 > 0.5, 1, 0)
M2tab <- table(M2_train_pred, train$DEATH_EVENT)

M2tab

M2_train_pred_accuracy <- sum(diag(M2tab)) / sum(M2tab)
M2tab ; M2_train_pred_accuracy                                  


#now for the test:


M2_test_output <- compute(model2, test[,-12])
M2temp <- M2_test_output$net.result
M2_test_pred <- ifelse(M2temp > 0.5, 1, 0)
M2_test_tab <- table(M2_test_pred, test$DEATH_EVENT)

M2_test_tab

M2_test_pred_accuracy <- sum(diag(M2_test_tab)) / sum(M2_test_tab)

M2_test_tab ; M2_test_pred_accuracy

# Pred accuracy in this case seems to be exactly the same, is it not affected by the number of hidden layers?
# drastically change hidden layers:



#---------------------

#model 3
#only us: time, sex, smoking and serum_creatinine, age, 



model3 <- neuralnet(
  DEATH_EVENT ~ age + serum_creatinine + sex + smoking + time,
  data = train, hidden = c(3), threshold = 0.01,
  linear.output = FALSE
)


plot(model3)


#first train

M3_train_output <- compute(model3, train[, -12])
M3 <- M3_train_output$net.result
M3_train_pred <- ifelse(M3 > 0.5, 1, 0)
M3_train_tab <- table(M3_train_pred, train$DEATH_EVENT)

M3_train_tab

M3_train_Pred_accuracy <- sum(diag(M3_train_tab)) / sum(M3_train_tab)

M3_train_tab ; M3_train_Pred_accuracy


#now test:

M3_test_output <- compute(model3, test[, -12])
M3temp <- M3_test_output$net.result
M3_test_pred <- ifelse(M3temp > 0.5, 1, 0)
M3_test_tab <- table(M3_test_pred, test$DEATH_EVENT)

M3_test_tab

M3_test_Pred_accuracy <- sum(diag(M3_test_tab)) / sum(M3_test_tab)

M3_test_tab ; M3_test_Pred_accuracy


