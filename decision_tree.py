import pandas as pd
import sklearn as sk
import graphviz
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score

#data perparation
data_glass_raw = pd.read_excel("C:/Users/Paulinka/Desktop/Paulina/Projects/Glass/Glass_data.xlsx")
data_glass_raw.head()
top = list(data_glass_raw.columns.values)[1:10]
data_glass = data_glass_raw.iloc[:,1:10]
labels = data_glass_raw.iloc[:,10:]

#data split
data_glass_train, data_glass_test, labels_train, labels_test= train_test_split(data_glass,labels, shuffle = True)

#first classifier
classifier = DecisionTreeClassifier(random_state = 0)
classifier.fit(data_glass_train, labels_train)
classes_names = classifier.classes_

#first tree
tree_chart = tree.export_graphviz(classifier, out_file=None,
...                      feature_names=top,
...                      class_names=classes_names,  
...                      filled=True, rounded=True,  
...                      special_characters=True)  
graph = graphviz.Source(tree_chart)  
graph 


#prediction
labels_prediction = classifier.predict(data_glass_test)

#Measuring Model Performance
#Accuracy
accuracy = classifier.score(data_glass_test, labels_test)
#Confusion_matrix
matrix = confusion_matrix(labels_test, labels_prediction)



confusion_matrix_plot = plot_confusion_matrix(classifier, data_glass_test, labels_test,
                        display_labels=classes_names,
                        cmap=plt.cm.RdPu,
                        xticks_rotation = "vertical")

#precision
precision_tree = precision_score(labels_test,labels_prediction, average='macro')


#maximum depth

maximum_depth_range = list(range(1,10))
accurancy_max_depth = []

for i in maximum_depth_range:
    best_classifier_depth = DecisionTreeClassifier(max_depth = i, random_state = 0)
    best_classifier_depth.fit(data_glass_train,labels_train)
    
    best_score = best_classifier_depth.score(data_glass_test,labels_test)
    accurancy_max_depth.append(best_score)

plot = plt.plot(maximum_depth_range,accurancy_max_depth)
plt.xlabel('Maximum Depth') 
plt.ylabel('Accurancy') 

best_accuracy_depth = max(accurancy_max_depth)
index = accurancy_max_depth.index(best_accuracy_depth) 
best_depth = maximum_depth_range[index]

#minimum samples leaf

minimum_samples_leaf = np.linspace(1,10,10, endpoint=True)
accurancy_minimum_samples_leaf = []

for i in minimum_samples_leaf:
    best_classifier_leaf = DecisionTreeClassifier(min_samples_leaf=int(i), random_state = 0)
    best_classifier_leaf.fit(data_glass_train,labels_train)
    
    best_score = best_classifier_leaf.score(data_glass_test,labels_test)
    accurancy_minimum_samples_leaf.append(best_score)   

plot = plt.plot(minimum_samples_leaf,accurancy_minimum_samples_leaf )
plt.xlabel('Minimum_samples_leaf') 
plt.ylabel('Accurancy') 

best_accuracy_leaf = max(accurancy_minimum_samples_leaf)

#best tree
best_tree_chart = tree.export_graphviz(best_classifier_leaf, out_file=None,
...                      feature_names=top,
...                      class_names=classes_names,  
...                      filled=True, rounded=True,  
...                      special_characters=True)  
graph = graphviz.Source(best_tree_chart)  
graph 

#prediction from best model
labels_prediction_best = best_classifier_leaf.predict(data_glass_test)
 
#Confusion matrix of best model
matrix_best = confusion_matrix(labels_test, labels_prediction_best)



confusion_matrix_plot = plot_confusion_matrix(best_classifier_leaf, data_glass_test, labels_test,
                        display_labels=classes_names,
                        cmap=plt.cm.RdPu,
                        xticks_rotation = "vertical")
#precision
precision_best = precision_score(labels_test,labels_prediction_best, average='micro')

#importance of feature
importance = pd.DataFrame({'Feature':data_glass_test.columns,'Importance': np.round(best_classifier_leaf.feature_importances_,3)})
importance.sort_values('Importance',ascending=False)
