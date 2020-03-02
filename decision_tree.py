import pandas as pd
import sklearn as sk
import graphviz
import matplotlib.pyplot as plt
import numpy as np
 
#data perparation
data_glass_raw = pd.read_excel("C:/Users/Paulinka/Desktop/Paulina/Projects/Glass/Glass_data.xlsx")
data_glass_raw.head()
top = list(data_glass_raw.columns.values)[1:10]
data_glass = data_glass_raw.iloc[:,1:10]
labels = data_glass_raw.iloc[:,10:]

#data split
from sklearn.model_selection import train_test_split
data_glass_train, data_glass_test, labels_train, labels_test= train_test_split(data_glass,labels, shuffle = True)

#first classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state = 0)
classifier.fit(data_glass_train, labels_train)
classes_names = classifier.classes_

#first tree
from sklearn import tree
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
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(labels_test, labels_prediction)
from sklearn.metrics import plot_confusion_matrix


confusion_matrix_plot = plot_confusion_matrix(classifier, data_glass_test, labels_test,
                        display_labels=classes_names,
                        cmap=plt.cm.RdPu,
                        xticks_rotation = "vertical")

#precision
from sklearn.metrics import precision_score
precision = precision_score(labels_test,labels_prediction, average='macro')

#model tuning

#maximum depth
maximum_depth_range = list(range(1,10))
accurancy_max_depth = []

for i in maximum_depth_range:
    best_classifier = DecisionTreeClassifier(max_depth = i, random_state = 0)
    best_classifier.fit(data_glass_train,labels_train)
    
    best_score = best_classifier.score(data_glass_test,labels_test)
    accurancy_max_depth.append(best_score)

plot = plt.plot(maximum_depth_range,accurancy_max_depth)
plt.xlabel('Maximum Depth') 
plt.ylabel('Accurancy') 

best_accuracy = max(accurancy_max_depth)
index = accurancy_max_depth.index(best_accuracy) 
best_depth = maximum_depth_range[index]

#minimum samples leaf

minimum_samples_leaf = np.linspace(1,10,10, endpoint=True)
accurancy_minimum_samples_leaf = []

for i in minimum_samples_leaf:
    best_classifier = DecisionTreeClassifier(min_samples_leaf=int(i), random_state = 0)
    best_classifier.fit(data_glass_train,labels_train)
    
    best_score = best_classifier.score(data_glass_test,labels_test)
    accurancy_minimum_samples_leaf.append(best_score)   

plot = plt.plot(minimum_samples_leaf,accurancy_minimum_samples_leaf )
plt.xlabel('Minimum_samples_leaf') 
plt.ylabel('Accurancy') 

best_accuracy = max(accurancy_minimum_samples_leaf)
