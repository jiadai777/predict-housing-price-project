
# coding: utf-8

# ## In this project, I will scale my previous prototype with Keras and Dask.

# In[1]:


import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd


# In[2]:


base = pd.read_csv('kc_house_data.csv')
base.head()


# In[3]:


X = base.iloc[:,[3,4,5,6,7,11,12,13,14,17,18]]
X.head()


# In[4]:


y = base.iloc[:,2]
y.head()


# In[5]:


X = X.values
y = y.values.reshape(-1,1)


# ## Scale the data

# In[6]:


get_ipython().run_cell_magic('time', '', 'from sklearn.preprocessing import StandardScaler\nscaler_X = StandardScaler()\nX = scaler_X.fit_transform(X)\nscaler_y = StandardScaler()\ny = scaler_y.fit_transform(y)')


# ## Split the dataset into training and testing

# In[7]:


from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=.2, random_state=0)


# In[8]:


# Import Keras modules
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error


# ## Build the network and fit the model in a loop with diferent learning rates
# Loss function values are stored in a dictionary for later plotting. MAEis also calculated with the testing dataset for each iteration.

# In[10]:


get_ipython().run_cell_magic('time', '', "dic_loss = {}\nlr_ = []\nmae_ = []\n\nlr_list = [.000001,.000005,.00001,.00005,.0001,.0005,.001,.005,.01,.05]\ncount = 0\n\nfor lr in lr_list:\n    \n    print('lr =',lr)\n    count += 1\n    print(str(count)+'/'+str(len(lr_list)))\n\n    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)\n    \n    model = Sequential()\n    model.add(Dense(6, input_dim=11, activation='relu'))\n    model.add(Dense(6, activation='relu'))\n    model.add(Dense(1, activation='linear'))\n    model.compile(loss='mse', optimizer=opt, metrics=['mse','mae'])\n    history = model.fit(X_treinamento, y_treinamento, epochs=100, verbose=0, batch_size=25)\n    \n    previsoes = model.predict(X_teste)\n\n    previsoes = scaler_y.inverse_transform(previsoes)\n    #y_teste = scaler_y.inverse_transform(y_teste)\n\n    mae = mean_absolute_error(scaler_y.inverse_transform(y_teste), previsoes)\n    \n    dic_loss[str(lr)] = history.history['loss']\n    lr_.append(lr)\n    mae_.append(mae)")


# CPU and Wall time are relatively long here. If this model is built with a much larger data (say 10 times bigger), we can imagine the time needed for this step will also be considerably longer.

# ## Plotting loss values and the MAE for each learning rate

# In[15]:


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

df = pd.DataFrame(dic_loss)
df.plot().grid()
plt.show()


# In[16]:


plt.semilogx(lr_, mae_)
plt.show()


# As can be seen, lr=0.001 is a good value for this problem, since the conversion is fast and the MAE is low.

# In[ ]:




