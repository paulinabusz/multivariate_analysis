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
from IPython.core.pylabtools import figsize
import seaborn as sns


def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns




#data perparation
data_glass_raw = pd.read_excel("C:/Users/Paulinka/Desktop/Paulina/Projects/Glass/Glass_data.xlsx")
top = list(data_glass_raw.columns.values)[0:9]
data_glass = data_glass_raw.iloc[:,0:10]

labels = data_glass_raw.iloc[:,10:]
features = data_glass.columns[:-1].tolist()


#Display top of dataframe
data_glass_raw.head()
data_glass_raw.describe()

#Missing values check
data_glass.info()
missing_values_table(data_glass)
data_glass['Type'].value_counts()


#distribution check

for feature in features:
    skew = data_glass[feature].skew()
    sns.distplot(data_glass[feature], label='Skew = %.2f' % (skew), kde= False, bins=30, color = 'blueviolet', )
    plt.legend(loc='upper right')
    plt.grid(color='darkgray', linestyle='--', linewidth=0.7)
    plt.show()

sns.kdeplot(data_glass['Type'], color = 'blueviolet')   

#detecting outliers - IDR range method 


for feature in features:
#first and third quartile    
    first_quartile = data_glass[feature].describe()['25%']
    third_quartile = data_glass[feature].describe()['75%']
#interquartile range
    iqr = third_quartile - first_quartile
    outliers = ((data_glass[feature] > (first_quartile - 3 * iqr)) &
            (data_glass[feature] < (third_quartile + 3 * iqr))) 
        
    print(outliers.value_counts())   
    
for feature in features:
    plt.figure(figsize=(5,5))
    sns.boxplot(data_glass[feature], orient = 'v', palette="Set3", linewidth=1)
    plt.show()    
        
#remove outliers for Na, Al, Si, K, Ca, Fe 1,3,4,5,6,8
features_outliers = data_glass[['RI','Na','Al','Si','K','Ca','Fe']]
features_outliers = list(features_outliers)

for feature_outliers in features_outliers:
#first and third quartile    
    first_quartile = data_glass[feature_outliers].describe()['25%']
    third_quartile = data_glass[feature_outliers].describe()['75%']
#interquartile range
    iqr = third_quartile - first_quartile
    data_glass = data_glass[(data_glass[feature_outliers] > (first_quartile - 3 * iqr)) &
            (data_glass[feature_outliers] < (third_quartile + 3 * iqr))]
    
#correlation
correlations_data = data_glass.corr()['Type'].sort_values()  
 
plt.figure(figsize=(8,8))
pallette = sns.diverging_palette(10, 220, sep=80, n=7)
sns.set_palette(pallette)
sns.pairplot(data_glass[features])
plt.show()


#data split
data_glass_train, data_glass_test, labels_train, labels_test= train_test_split(data_glass.iloc[:,0:9],data_glass.iloc[:,-1], shuffle = True)

#first classifier
classifier = DecisionTreeClassifier(random_state = 0)
classifier.fit(data_glass_train, labels_train)
classes_names = str(classifier.classes_)

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
