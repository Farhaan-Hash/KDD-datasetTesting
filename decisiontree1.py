from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold


#Importing Training Dataset
df1 = pd.read_csv("Input_Train11.csv")
df2 = pd.read_csv("Output_Train11.csv")

#input
x = np.array(df1)

#output
y = np.array(df2)

#Divivding dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20) 


#Using Decision Tree with Gini Index
clf_gini = DecisionTreeClassifier(criterion = "gini",max_depth=10,random_state=100)
clf_gini.fit(X_train, y_train)

#Using Decision Tree with Gini Index Output
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10,max_features=None, max_leaf_nodes=None,
min_samples_split=2, min_weight_fraction_leaf=0.0,presort=False, random_state=100,splitter='best')

#Decision Tree with Information Gain
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=10)
clf_entropy.fit(X_train, y_train)

#Decision Tree with Information Gain Output
DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=10,max_features=None, max_leaf_nodes=None,min_samples_split=2, min_weight_fraction_leaf=0.0,presort=False, random_state=100, splitter='best')


#validation
kf = KFold(n_splits=5) 
kf.get_n_splits(x)

print(kf)  
KFold(n_splits=5, random_state=None, shuffle=False)
for train_index, test_index in kf.split(x):
   print("TRAIN:", train_index, "TEST:", test_index)
X_train, X_test = x[train_index], x[test_index]
y_train, y_test = y[train_index], y[test_index]




#Gini Index Prediction for Test dataset
y_pred = clf_gini.predict(X_test)
print(y_pred)

#Information Gain Decision Tree Prediction for Test dataset
y_pred_en = clf_entropy.predict(X_test)
print(y_pred_en)

#Gini Index Accuracy
Accuracy_Gini = accuracy_score(y_test,y_pred)*100
print(Accuracy_Gini)
        
#Information Gain Accuracy
Accuracy_en = accuracy_score(y_test,y_pred_en)*100
print(Accuracy_en)
        

cm=confusion_matrix(y_test,y_pred) 
print(cm)
print(classification_report(y_test,y_pred))


print("False Positive rate: "+ str(cm[1,0]/(cm[0,0]+cm[1,0])))
Accuracy = accuracy_score(y_test,y_pred)
print("Accuracy: %.2f%%" % (Accuracy * 100.0))

print('Precision: ' + str((cm[0,0]/(cm[0,0]+cm[1,0]))))
print('Recall: ' + str((cm[0,0]/(cm[0,0]+cm[0,1]))))



#plotting results
plt.title('Decision Tree')
plt.plot(X_test,y_pred ,color='b',marker='o',markerfacecolor='m',linestyle='--',linewidth=1)
plt.xlabel("Test Data")
plt.ylabel("Predicted Attacks")
plt.savefig('DT.png',dpi=500)
plt.show()



