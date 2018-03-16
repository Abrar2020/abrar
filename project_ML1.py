
# coding: utf-8

# In[273]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib
import sklearn
# machine learning library 
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
# naive bayse aogorithms 
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn import metrics 


# In[274]:


get_ipython().magic('cd "C:\\Users\\sony\\Desktop"')


# In[275]:


df=pd.read_csv("C:/Users/sony/Desktop/machine_learning/xAPI-Edu-Data-updated.csv")


# In[276]:


df.head(1)


# In[277]:


df.tail(1)


# In[278]:


df.shape
# number of Features and 


# In[279]:


x=df.iloc[:,0:17]

# show the 17 features and from Paper we try to select behaviours variables 
print (x)


# In[280]:


y=df.iloc[:,-1]
# show class part of data 


# In[197]:


print (y)


# In[281]:


x_train, x_test, y_train, y_test= train_test_split (x,y,test_size=.33, random_state=17)


# In[282]:


from sklearn import preprocessing


# In[283]:


le = preprocessing.LabelEncoder()


# In[284]:


le.fit(y)


# In[285]:


le.classes_


# In[286]:


t=le.transform(y)
# transform y  to int 
# H to 0 , L to 1 , M to 2


# In[287]:


print (t)


# In[288]:


GausNB=GaussianNB()


# In[289]:


x_train


# In[290]:


x_test


# In[291]:


y_train


# In[292]:


y_test


# In[293]:


df.groupby('Class').sum()
# check the acountable features 


# In[294]:


from sklearn import preprocessing
import numpy as np


# In[295]:


df2 = preprocessing.LabelEncoder()


# In[296]:


dfM = df[df.gender == 'M']


# In[297]:


df2=pd.read_csv("C:/Users/sony/Desktop/machine_learning/xAPI-Edu-Data.csv")


# In[298]:


le = preprocessing.LabelEncoder()


# In[299]:


dfM.groupby('Class').sum()
# Count the 4 important features for Female 


# In[300]:


dfF = df[df.gender == 'F']
dfFH=df[df.Class == 'H']


# In[301]:


dfF.VisITedResources.mean()


# In[302]:


dfFH.AnnouncementsView.mean()


# In[303]:


dfF.groupby('Class').sum()
# count the 4 important Features for Male 


# In[304]:


def removedups():
    return set(df)

print(removedups())


# In[305]:


df.shape


# In[306]:


def removedups():
    return set(x)

print(removedups())


# In[307]:


#df['TotalQ'] = df['Class']
# declare object of totalQ
#print(df['TotalQ'])

#df['TotalQ'].loc[df.TotalQ == 'Low-Level'] = 0.0
#df['TotalQ'].loc[df.TotalQ == 'Middle-Level'] = 1.0
#df['TotalQ'].loc[df.TotalQ == 'High-Level'] = 2.0
#print (df['TotalQ'])
#print (df)


# In[308]:



continuous_subset = df.ix[:,9:13]
print (continuous_subset)
# Declare behavioral variables 


# In[309]:


continuous_subset['gender'] = np.where(df['gender']=='M',1,0)
# add the gender 
continuous_subset['Parent'] = np.where(df['Relation']=='Father',1,0)
# add the parent 
print (continuous_subset)
print (continuous_subset['Parent'])
X = np.array(continuous_subset).astype('float64')
print (X)
# assign x with the most important Features of train data


# In[310]:


#y = np.array(df['TotalQ'])
y=np.array(df['Class'])
# the cllasification after converted 
X.shape

# filter with number of features 
# all of data sets (raisedhands , VisITedResources , AnnouncementsView , Discussion,Gender ,Parent


# In[311]:


from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
# for classification and comparision and Preprocessing before svm algorithms
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
# Define the object from standard scale 
sc.fit(X_train)

X_train_std = sc.transform(X_train)
# use transform function of standard Scale 
X_test_std = sc.transform(X_test)
# transfer number to mean and variance 


# In[312]:


from sklearn.linear_model import Perceptron
# use before classification and comparision and Preprocessing before svm algorithms
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
# intialize the random variable and eta 
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())


# In[313]:


from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


# In[230]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[314]:


from sklearn.svm import SVC
# Support Vector Classification 
svm = SVC(kernel='linear', C=6.0, random_state=0)
# http://scikit-learn.org/stable/modules/svm.html
svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())


# In[315]:


print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


# In[316]:


print(classification_report(y_test, y_pred))


# In[317]:


df.Class.value_counts()


# In[318]:


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

svm = SVC(kernel='linear', C=3.0, random_state=0)
svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


# In[323]:


'''cross validation'''
clf = SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y, cv=10)
scores 
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


from sklearn.linear_model import Perceptron

ppn = Perceptron(n_iter=100, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print('Misclassified percentage samples: %d' % (y_test != y_pred).sum())


'''cross validation'''
scores = cross_val_score(ppn, X, y, cv=10)
scores 
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[325]:


svm = SVC(kernel='rbf', random_state=0, gamma=2, C=2.0)
#  SVM with the Radial Basis Function (RBF) kernel, Each 
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())


# In[320]:


print (y_pred)


# In[164]:


import matplotlib.pyplot as plt
plt.plot(y_pred)
plt.show()


# In[326]:


#print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('The Accuracy is %.2f'%  accuracy_score(y_test, y_pred))


# In[327]:


df.groupby('Class').describe()


# In[328]:


X.shape


# In[268]:


print(classification_report(y_test, y_pred))


# In[329]:


import seaborn as fnf
fnf.countplot(x='StudentAbsenceDays',data = df, hue='Class',palette='bright')
plt.show()
# itis important featues 


# In[330]:


continuous_subset['Absences'] = np.where(df['StudentAbsenceDays']=='Under-7',0,1)
# add the absent Features , 
X = np.array(continuous_subset).astype('float64')
y = y=np.array(df['Class'])
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=0)
sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[331]:


svm = SVC(kernel='rbf', random_state=0, gamma=2, C=2.0)
#  SVM with the Radial Basis Function (RBF) kernel, Each 
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())


# In[332]:


print('The Accuracy is %.2f'%  accuracy_score(y_test, y_pred))


# In[245]:


X.shape
# all of data sets (raisedhands , VisITedResources , AnnouncementsView , Discussion,Gender ,Parent ,appsent days


# In[333]:


continuous_subset['parentssatifaction'] = np.where(df['ParentschoolSatisfaction']=='Good',1,0)
X = np.array(continuous_subset).astype('float64')
y = np.array(df['Class'])
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=0)
sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[334]:


svm = SVC(kernel='rbf', random_state=0, gamma=2, C=2.0)
#  SVM with the Radial Basis Function (RBF) kernel, Each 
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
# all of data sets (raisedhands , VisITedResources , AnnouncementsView , Discussion,Gender ,Parent ,appsent days ,
# the parent satisfication not impotant 


# In[335]:


print('The Accuracy is %.2f'%  accuracy_score(y_test, y_pred))


# In[94]:


continuous_subset['ParentAnsweringSurvey'] = np.where(df['ParentAnsweringSurvey']=='Yes',1,0)


# In[336]:


X = np.array(continuous_subset).astype('float64')
y = np.array(df['Class'])
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=0)
sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[337]:


print('The Accuracy is %.2f'%  accuracy_score(y_test, y_pred))


# In[182]:


svm = SVC(kernel='rbf', random_state=0, gamma=2, C=2.0)
#  SVM with the Radial Basis Function (RBF) kernel, Each 
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())


# In[183]:


print('The Accuracy is %.2f'%  accuracy_score(y_test, y_pred))
# the parent parent behavior not  not impotant 


# In[246]:


df['Topic'].loc[df.Topic == 'IT'] = 0
df['Topic'].loc[df.Topic == 'Math'] = 1
df['Topic'].loc[df.Topic == 'Arabic'] = 2
df['Topic'].loc[df.Topic == 'Science'] = 3
df['Topic'].loc[df.Topic == 'English'] = 4
df['Topic'].loc[df.Topic == 'Quran'] = 5
df['Topic'].loc[df.Topic == 'Spanish'] = 6
df['Topic'].loc[df.Topic == 'French'] = 7
df['Topic'].loc[df.Topic == 'History'] = 8
df['Topic'].loc[df.Topic == 'Biology'] = 9
df['Topic'].loc[df.Topic == 'Chemistry'] = 10
df['Topic'].loc[df.Topic == 'Geology'] = 11


# In[247]:


continuous_subset['Topic']=df['Topic']


# In[248]:


X = np.array(continuous_subset).astype('float64')
y = np.array(df['Class'])
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=0)
sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[249]:


svm = SVC(kernel='rbf', random_state=0, gamma=2, C=2.0)
#  SVM with the Radial Basis Function (RBF) kernel, Each 
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())


# In[338]:


print('The Accuracy is %.2f'%  accuracy_score(y_test, y_pred))


# In[339]:


X.shape

