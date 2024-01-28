#!/usr/bin/env python
# coding: utf-8

#  ![WhatsApp%20Image%202024-01-24%20at%202.06.04%20PM.jpeg](attachment:WhatsApp%20Image%202024-01-24%20at%202.06.04%20PM.jpeg)

# # ................... Adult UCI Cencus Income Analysis and Prediction ................¶

# ## ........................................................ Data Loading .......................................................

# ### Importing Libraries 

# In[1]:


# Basic Libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px 
import warnings 
warnings.filterwarnings("ignore")

# Machine Learning Algorithm libraries 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Feature engenering and training model libraries
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ### Reading CSV file 

# In[2]:


rohan_df = pd.read_csv("Adult.csv")
rohan_df 


# ### Exploring Data 

# In[3]:


rohan_df.head()


# In[4]:


rohan_df.tail()


# In[5]:


rohan_df.shape


# In[6]:


rohan_df.isnull().sum()


# In[7]:


rohan_df.info()


# In[8]:


rohan_df.nunique()


# ### Value Count Function  

# In[9]:


rohan_df["workclass"].value_counts()


# In[10]:


rohan_df["native-country"].value_counts()


# In[11]:


rohan_df["occupation"].value_counts()


# In[12]:


rohan_df["marital-status"].value_counts()


# In[13]:


rohan_df["gender"].value_counts()


# ### Filling the "?"  Values With Mode

# In[14]:


rohan_df["workclass"] = rohan_df["workclass"].replace("?" , "Private") 
rohan_df["occupation"] = rohan_df["occupation"].replace("?" , "Prof-specialty") 
rohan_df["native-country"] = rohan_df["native-country"].replace("?" , "United-States") 


# In[15]:


rohan_df.head(10)


# In[16]:


rohan_df["income"].value_counts()


# In[17]:


rohan_df["education"].value_counts()


# ### Feature Engineering 

# In[18]:


#education Category
rohan_df.education = rohan_df.education.replace(["Preschool" , "1st-4th" , "5th-6th" , "7th-8th" , "9th" , "10th" ,"11th" ,"12th"] , "left")
rohan_df.education = rohan_df.education.replace("HS-grad" , "school")
rohan_df.education = rohan_df.education.replace(["Assoc-voc" , "Assoc-acdm" , "Prof-school" , "Some-college"] , "higher")
rohan_df.education = rohan_df.education.replace("Bachelors" , "undergrad")
rohan_df.education = rohan_df.education.replace("Masters" , "grad")
rohan_df.education = rohan_df.education.replace("Doctorate" , "doc")


# In[19]:


#Marital Status 
rohan_df["marital-status"] = rohan_df["marital-status"].replace(["Married-civ-spouse", "Married-AF-spouse"] , "married")
rohan_df["marital-status"] = rohan_df["marital-status"].replace( "Never-married" , "not married") 
rohan_df["marital-status"] = rohan_df["marital-status"].replace(["Never-married", "Divorced" , "Separated" , "Widowed" , "Married-spouse-absent" ] , "not married")


# In[20]:


#income
rohan_df.income = rohan_df.income.replace("<=50K" , 0)
rohan_df.income = rohan_df.income.replace(">50K" , 1)


# ## ............................................ Data Visualization and Analysis .........................................

# ### Histogram  

# In[21]:


rohan_df.hist(figsize=(12,12) , layout= (3,3) , sharex = False)


# ### Boxplot

# In[22]:


rohan_df.plot(kind="box" , figsize = (12,12) , layout = (3,3) , sharex = False , subplots = True)


# ### Pie Chart 

# In[23]:


px.pie(rohan_df , values="educational-num" , names = "education" , title = "Percentage of Education")


# ### Scatter Plot  

# In[24]:


px.scatter(rohan_df , x = "capital-gain" , y = "hours-per-week" , color = "gender" , title = "Scatter plot Capital-gain and Hours-per-Week as per Gender")


# ### Pie Chart

# In[25]:


px.pie(rohan_df , values = "hours-per-week" , names = "occupation")


# ## .................................. Preparing the Data for Model ........................................ 

# ### Feature Scaling 

# In[26]:


from sklearn.preprocessing import StandardScaler , LabelEncoder
rohan_df1 = rohan_df.copy()
rohan_df1 = rohan_df1.apply(LabelEncoder().fit_transform)
rohan_df1.head()


# In[27]:


from sklearn.preprocessing import StandardScaler , LabelEncoder
ss = StandardScaler().fit(rohan_df1.drop("income" , axis = 1))
X = ss.transform(rohan_df1.drop("income" , axis = 1))
y = rohan_df["income"]


# ### Prepare Out data for Traning

# In[28]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split( X,y,test_size = 0.3 , random_state = 40 )


# In[29]:


# After the Scaling Data
rohan_df.head()


# ### Check Sample Size 

# In[30]:


print("X_train :", X_train.size)
print("X_test :", X_test.size)
print("y_train :" ,y_train.size)
print("y_test :", y_test.size)


# ## .......................................... Applying Machine Learning Algorithm ..........................................

# ### Machine Learning Algorithm : Logistic Regression 

# In[31]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

lr = LogisticRegression()
model1 = lr.fit(X_train , y_train)
Prediction1 = model1.predict(X_test)

print( "Testing Accurancy :" , accuracy_score(y_test , Prediction1))


# ### Machine Learning Algorithm : KNeighborsClassifier 

# In[32]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=3)
model2=KNN.fit(X_train,y_train)
Prediction2 = model2.predict(X_test)


from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,Prediction2))


# ### Machine Learning Algorithm : SVC

# In[33]:


from sklearn.svm import SVC
SVC = SVC()
model3 = SVC.fit(X_train,y_train)
Prediction3 = model3.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,Prediction3))


# ### Machine Learning Algorithm : DecisionTreeClassifier

# In[34]:


from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
model4 = DT.fit(X_train,y_train)
Prediction4 = model4.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,Prediction4))


# ###  Machine Learning Algorithm : GaussianNB

# In[35]:


from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
model5 = GNB.fit(X_train,y_train)
Prediction5 = model5.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,Prediction5))


# ### Machine Learning Algorithm : RandomForestClassifier

# In[36]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
model6 = RF.fit(X_train, y_train)
Prediction6  = model6.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,Prediction6))


# In[37]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['Log-Reg', 'KNN', 'SVC', 'Des-Tree', 'Gaus-NB', 'RandomForest']
accuracy = [84.27 , 82.72 ,85.10, 81.38 , 82.41 , 85.67 ]
ax.bar(langs,accuracy)
plt.show()


# ### The Best Accuracy is given by RandomForest classifier is 85.62 . 

# In[38]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , Prediction6)
cm


# In[39]:


sns.heatmap(cm , annot = True , cmap = "BuPu")
plt.show()


# ### precision and recall of the model

# In[40]:


from sklearn.metrics import classification_report
print( classification_report(y_test , Prediction6))


# ### Saving the Decision tree model for future prediction 

# In[41]:


import pickle
filename = 'final_model.sav'
pickle.dump(model6 , open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result,'% Acuuracy')


# ### The Best Accuracy is given by RandomForest classifier is 85.62 . Hence we will use RandomForest classifier  algorithms for training my model
