#Import Required Libraries
import pandas as pd,seaborn as sb,matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


#load the Dataset
data=pd.read_csv("creditcard.csv")


#Explore, Process the Dataset
print("Number of Columns: ",len(data.columns))
print()
print("Columns in Data: ")
print(data.columns)
print()
print("Shape of data: ",data.shape)
print()
print("Description of data: ")
print(data.describe())


#Take 10% of data as Large data
data=data.sample(frac=0.1,random_state=1)
print("New size of data: ",data.shape)


#Histograms
print()
print("Histograms: ")
data.hist(figsize=(20,20))
plt.show()


#Check no. of Fraud
fraud=data[data['Class']==1]
valid=data[data['Class']==0]
print("Fraud: ",len(fraud))
print("Valid: ",len(valid))
fraud_frac= len(fraud)/float(len(valid))
print("Fraction fraud: ",fraud_frac)
print()


#Correlation matrix, Heatmap
corrmat=data.corr()
fig=plt.figure(figsize = (12, 9))
print("Heatmap: ")
sb.heatmap(corrmat,vmax=.8,square=True)
plt.show()


#Data Formatting 
cols=data.columns.tolist()
cols=[i for i in cols if i not in ["Class"]]
target="Class"
X=data[cols]
Y=data[target]
print("Size of X (data except target): ",X.shape)
print("Size of target: ",Y.shape)
print()
print(80*"-")


#MODEL FITTING
print("Model fitting: ") 

#This is done by two different Models- 
classifiers={"Isolation Forest":IsolationForest(max_samples=len(X),contamination=fraud_frac,random_state=1),
             "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20,contamination=fraud_frac)}
no_of_fraud=len(fraud)


#Fitting each Model
for i, (clf_name, clf) in enumerate(classifiers.items()):
    # Fit the data and tag frauds
    if clf_name=="Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    #the above returns -1 for fraud and 1 for other so,
    # Reshape the prediction values to 0 for valid, 1 for fraud.
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    no_of_errors = (y_pred != Y).sum()
    
    # Run the Classification metrics such as accuracy
    print("For: ",clf_name," ","no. of fraud detected: ",no_of_errors)
    print()
    print("Accuracy Score: ",accuracy_score(Y, y_pred))
    print()
    print("Classification Report: ")
    print(classification_report(Y, y_pred))