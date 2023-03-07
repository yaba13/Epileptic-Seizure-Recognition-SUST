import pandas as pd
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import lime
from lime import lime_tabular
warnings.filterwarnings('ignore')
from tensorflow.keras.layers import LSTM
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
# ## RNN (Neural Network)

# ### feature engineering for RNN

# In[ ]:


y = to_categorical(y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train = x_train
X_test = x_test

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_train = np.reshape(x_train, (x_train.shape[0],1,X.shape[1]))

x_test = scaler.fit_transform(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],1,X.shape[1]))



# In[ ]:


print(type(x_train))
print(type(x_test))
print(type(X_train))
print(type(X_test))
print(type(y_train))
print(type(y_test))



# ## Building Models (classical ML)
# ### LSTM / BiLSTM

# In[ ]:


tf.keras.backend.clear_session()

model = Sequential()
model.add(LSTM(64, input_shape=(1,4096),activation="relu",return_sequences=True))
model.add(LSTM(32,activation="sigmoid"))
model.add(Dense(2, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])
model.summary()


# ### Fitting Model

# In[ ]:


history = model.fit(x_train, y_train, epochs = 10)


# ### Accuracy Evaluation

# Train data
# 

# In[ ]:


scoreTrain, accTrain = model.evaluate(x_train, y_train)
print(round(accTrain*100, 2), '%')


# Test Data

# In[ ]:


scoreTest, accTest = model.evaluate(x_test, y_test)
print(round(accTest*100, 2), '%')





# **Individual check**

# In[ ]:


#print(x_test[22,:])
#print(y_test[22,:])


# In[ ]:


#scoreTest, accTest = model.evaluate(x_test[[44],:], y_test[[44],:])
#print(round(accTest*100, 2), '%')



# In[ ]:


#print(model.predict(x_test[[44],:]))


# ***Model interpretation (LIME)***

# In[ ]:


explainer = lime_tabular.RecurrentTabularExplainer(training_data = x_train, 
                                                   feature_names = X_train.columns,
                                                   class_names = [0, 1],
                                                   mode='classification'
                                                   )

exp = explainer.explain_instance(np.array(X_test.iloc[123]), model.predict)
exp.show_in_notebook(show_table=True)
from sklearn.metrics import confusion_matrix
trainpredict = model.predict(x_train)
testpredict = model.predict(X_test)
cm = confusion_matrix(y_test,testpredict )
print(cm)

