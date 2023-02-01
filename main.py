import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler




st.write('''# Hello world ''')

st.write('''# PCA (Principle Component Analysis) is beautiful idea to visualize in 2d or 3d
##### Today we are going to perforn PCA without using sklearn
##### PCA is mechanism that totally depends on the Variance. it find the axes that 
##### maximize the varaince 
##### we will perform PCA on MNIST Dataset which is 784-d data we project to 2-d using PCA 
# Code''')

code = '''import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

x = pd.read_csv("train.csv")
y=x["label"]
x.drop("label",axis=1,inplace=True)

labels = y.head(15000)
data = x.head(15000)


standardized_data = StandardScaler().fit_transform(data)
covar_matrix = np.matmul(standardized_data.T , standardized_data) #covariance
values, vectors = eigh(covar_matrix, eigvals=(782,783))
data_2d = np.matmul(standardized_data,vectors)

dataframe = pd.DataFrame({"1st_principal":data_2d[:,:1].flatten(),"2nd_principal":data_2d[:,1:].flatten(),"label":labels})
warnings.filterwarnings('ignore')
fig=plt.figure()
sns.scatterplot(data=dataframe,hue="label",x="1st_principal",y="2nd_principal",palette=sns.color_palette("husl", 10))
'''

st.code(code,language="python")
st.write('''# Output in 2d''')

x = pd.read_csv("train.csv")
y=x["label"]
x.drop("label",axis=1,inplace=True)

labels = y.head(15000)
data = x.head(15000)
standardized_data = StandardScaler().fit_transform(data)
covar_matrix = np.matmul(standardized_data.T , standardized_data)
values, vectors = eigh(covar_matrix, eigvals=(782,783))
data_2d = np.matmul(standardized_data,vectors)
dataframe = pd.DataFrame({"1st_principal":data_2d[:,:1].flatten(),"2nd_principal":data_2d[:,1:].flatten(),"label":labels})
fig=plt.figure()
sns.scatterplot(data=dataframe,hue="label",x="1st_principal",y="2nd_principal",palette=sns.color_palette("husl", 10))
st.pyplot(fig)



