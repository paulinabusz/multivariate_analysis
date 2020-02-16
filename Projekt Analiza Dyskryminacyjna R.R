dane<-Data_bankrupcy
dane[,11]<-as.factor(unlist(dane[,11]))
grupy<-as.factor(unlist(dane[,11]))

#Package psych
#Package devtools
install_github("fawda123/ggord")
#Package klaR

#szukamy kombinacji liniowej tych 8 zmiennych, która w najlepszy sposób podzieli zmienne na dwie grupy

library(psych)



pairs.panels(dane[1:5],gap=0,bg=c("pink","blue")[dane$Class],pch=21)
pairs.panels(dane[6:10],gap=0,bg=c("pink","blue")[dane$Class],pch=21)

#Data partition

set.seed(555)

#Ustawiam ziarno na 555, tzn funkcja set.seed wygeneruje z tego ziarna pseudolosowa liczbe.

ind<-sample(2,nrow(dane),
            replace = TRUE,
            prob = c(0.6,0.4)) 

#rozdzielam zbiór danych na dwie grupy, z czego jedna zawiera 60% obserwacji a druga 40%.

training<-dane[ind==1,]  
testing<-dane[ind==2,]

#Linear discriminant analysis
library(MASS)
linear<-lda(Class~.,training)
linear
attributes(linear)
linear$counts

#Histogram


p<-predict(linear, training)
p # $class jest predykcja na temat tego czy dana osoba bedzie cukrzykiem czy tez nie, $posterior prawdopodobienstwo, ze dana osoba bedzie cukrzykiem.
attributes(p)
p$class
ldahist(data=p$x[,1],g=training$Class)


#Partition plot
library(klaR)
library(devtools)
partimat(Class~., data = training, method = "lda")

#Confusion matrix and accurancy - training data
p1<-predict(linear,training)$class
tab<-table(Predicted = p1, Actual = training$Class)
tab
sum(diag(tab))/sum(tab)

#Confusion matrix and accurancy - testing data
p2<-predict(linear, testing)$class
tab2<-table(Predicted = p2, Actual = testing$Class)
tab2
sum(diag(tab2))/sum(tab2)

########## Estymator Bayesowski ##########

library(e1071)

#Esytmator bayesowski dla grupy uczacej

grupy_training<-as.factor(unlist(training[,11]))
estymator_bayesowski_training<-naiveBayes(grupy_training ~ ., data = training)

#Predykcja dla grupy uczacej

p_bayes_training<-predict(estymator_bayesowski_training, training)
p # $class jest predykcja na temat tego czy dana osoba bedzie cukrzykiem czy tez nie, $posterior prawdopodobienstwo, ze dana osoba bedzie cukrzykiem.
attributes(p)
p$class

#Macierz kontyngencji dla grupy uczacej

p_bayes_training_1<-predict(estymator_bayesowski_training,training, type=("class"))
tab_training<-table(Predicted = p_bayes_training_1, Actual = training$Class)
tab
sum(diag(tab_training))/sum(tab_training)

#Estymator Bayesowski dla próby testujacej

grupy_testing<-as.factor(unlist(testing[,11]))
estymator_bayesowski_testing<-naiveBayes(grupy_testing ~ ., data = testing)

#Predykcja dla grupy testujacej

p_bayes_testing<-predict(estymator_bayesowski_testing, testing)
p # $class jest predykcja na temat tego czy dana osoba bedzie cukrzykiem czy tez nie, $posterior prawdopodobienstwo, ze dana osoba bedzie cukrzykiem.
attributes(p)
p$class

#Macierz kontyngencji dla grupy testujacej

p_bayes_testing_1<-predict(estymator_bayesowski_testing,testing, type=("class"))
tab_testing<-table(Predicted = p_bayes_testing_1, Actual = testing$Class)
tab_testing
sum(diag(tab_testing))/sum(tab_testing)

#wektory nosne 

library(e1071)

#wektory nosne dla próby uczacej

model_training<-svm(grupy_training~., data=training)
summary(model_training)

p_training_svm<-predict(model_training,training, type=("class"))
tab_training_svm<-table(Predicted = p_svm, Actual = training$Class)
tab_training_svm
sum(diag(tab_training_svm))/sum(tab_training_svm)

#wektory nosne dla próby testujacej

model_testing<-svm(grupy_testing~., data=testing)
summary(model_testing)

p_testing_svm<-predict(model_testing,testing, type=("class"))
tab_testing_svm<-table(Predicted = p_testing_svm, Actual = testing$Class)
tab_testing_svm
sum(diag(tab_testing_svm))/sum(tab_testing_svm)
