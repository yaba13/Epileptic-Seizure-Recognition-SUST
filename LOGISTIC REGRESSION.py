import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import lime
from lime import lime_tabular
warnings.filterwarnings('ignore')


# ### Dataframe Enquiry
# 

# In[ ]:


df = pd.read_csv(r'C:/Users/hp/Desktop/(final).f.csv')
df


# ### Feature Label extraction

# In[ ]:


X = df.iloc[:,1:-1]
y = df.iloc[:,-1:]


# Feature

# In[ ]:


X


# Label
# 

# In[ ]:


y


# ### Feature Engineering

# #### Multi label to binary label conversion

# In[ ]:


def toBinary(x):
    if x != 1: return 0;
    else: return 1;


# In[ ]:


y = y['y'].apply(toBinary)
y = pd.DataFrame(data=y)
y


# #### Feature Scaling

# In[ ]:


# scaler = StandardScaler()
# X = scaler.fit_transform(X)


# In[ ]:


X


# In[ ]:


y


# ### Splitting Data (Train-Test)

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[ ]:


x_test.iloc[1]


# ## Building Models (classical ML)

# ### Logistic Regression

# **Training data accuracy evaluation**

# In[ ]:


clf = LogisticRegression() #initializing logistic regression
clf.fit(x_train, y_train) #training the model with train data(input, output)
acc_log_reg = clf.score(x_train, y_train) * 100
print(round(acc_log_reg,2), "%")


# **Test Data accuracy evaluation**

# In[ ]:


y_pred_log_reg = clf.predict(x_test)
acc_log_reg2 = round(clf.score(x_test, y_test) * 100, 2)
print(acc_log_reg2, "%")


# ***Model Report***
# 

# In[ ]:


predictions = clf.predict(x_test)
print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predictions)
print(cm)


# ***Model interpretation (LIME)***

# In[ ]:


explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(x_train),
    feature_names=x_train.columns,
    class_names=[0, 1],
    mode='classification'
)


# In[ ]:


exp = explainer.explain_instance(
    data_row=x_test.iloc[0], 
    predict_fn=clf.predict_proba
)

exp.show_in_notebook(show_table=True)
