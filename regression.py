# import libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from streamlit.proto.Slider_pb2 import Slider
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from yellowbrick.regressor import PredictionError
from streamlit_yellowbrick import st_yellowbrick





# Part 1: Build project
data = pd.read_csv('data_newest.csv')

# Data preprocessing
data = data.reset_index()
data = data.dropna()
data = data.drop_duplicates()
data_new = data[['channel','dim', 'weight', 'retail', 'selling']]

x = data_new[['retail', 'dim','weight', 'channel']]
y = data_new['selling']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Scale data

# Build model
model = ensemble.GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_leaf=15, min_samples_split=10, learning_rate=0.01, loss='huber', random_state=5)
model.fit(x_train, y_train)

# Evaluation
y_pred = model.predict(x_test)
predict_df = pd.DataFrame({"y_test" : y_test, "y_predict" : y_pred})
mse= mean_squared_error(predict_df.y_predict, predict_df.y_test)
rmse= np.sqrt(mean_squared_error(predict_df.y_predict, predict_df.y_test))
r2 = r2_score(predict_df.y_predict, predict_df.y_test)
mae = mean_absolute_error(predict_df['y_test'], predict_df['y_predict'])

# channel_dict = {'Non-DI': 0, 'DI': 1}
# Part 2: Show project's result with Streamlit

# Hiển thị
# CURRENT_THEME = "green"
st.image("https://user-images.githubusercontent.com/67365937/149266024-d200c0e5-1d61-4f5d-b49d-78865a27e8e5.png",width=250)
st.write('# Supply Chain Planning Team')
st.write("### Sell-in Price Prediction Project")





# menu = ['Overview', 'Build Project', 'New Prediction']
# choice = st.sidebar.selectbox('Menu', menu)
# if choice == 'Overview':
#     st.subheader('Overview')
#     st.write("""
#     #### The data has been split into two groups:
#     - training set (train.csv):
#     The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.
#     - test set (test.csv):
#     The test set should be used to see how well your model performs on unseen data. 
#     """)
# elif choice == 'Build Project':
#     st.subheader('Build Project')    
#     st.write('#### Data Preprocessing')    
#     st.write('#### Show data:')
#     st.table(data_new.head(5))

#     st.write('#### Build model and evaluation')      
#     st.write('Mean squared error: {}'.format(round(mse,2)))  
#     st.write('Root mean squared error: {}'.format(round(rmse,2)))  
#     st.write('Mean absolute error: {}'.format(round(mae,2))) 
#     st.write('R squared: {}'.format(round(r2,2))) 
#     st.table(predict_df.head(10))      
    

    # st.write('#### Visualization')
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111)
    
    # visualizer = PredictionError(model)
    # visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
    # visualizer.score(x_test, y_test)  # Evaluate the model on the test data
    # visualizer.ax.set_title("Prediction Error for Gradient Boosting Regressor", fontsize= 20, fontweight = 'bold')

    # visualizer.ax.set_xlabel("Actual selling", fontsize= 18)
    # visualizer.ax.set_ylabel("Predicted selling", fontsize= 18)
    # visualizer.finalize()
    # xpoints = ypoints = plt.xlim()
    # plt.plot(xpoints, ypoints, linestyle='--', color='r', lw=3, scalex=False, scaley=False)
    # plt.show()
    
    # st_yellowbrick(visualizer)     

# elif choice == 'New Prediction':
#     st.subheader('Make new prediction')
st.write('##### Input/Select data')
name = st.text_input('Product Name')
    
channel = st.selectbox('DI Channel: 1, Non-DI Channel: 0 ', options=[1, 0])
retail = st.number_input('Insert Retail Price ($)')
weight = st.number_input('Insert Weight (lbs)')
dim = st.number_input('Insert Dim_Weight (cm3)')
# color = st.selectbox('Insert Color', options=['Red','Blue','Green','Yellow','Pink','Black'])
# size = st.selectbox('Insert Size', options=['Small','Medium','Large'])
# cat = st.selectbox('Insert Category', options=['Ball','Kettelbell','Dumbbell', 'AB Mat','Roller EPP'])
    # retail = st.slider('Retail ($)', 0.2,600.0,0.01)
    # weight = st.slider('Weight (lbs)', 0.0, 150.0, 0.01)
    # dim = st.slider('Dimension (inch3)', 0.0, 16000.0, 1.0)
    
    # make new prediction
st.subheader('Model Result')
new_data_1 = [[retail, dim, weight, channel]]
prediction = model.predict(new_data_1)
min7 = prediction*0.94 if channel == 0 else prediction*0.8
max7 = prediction*1.35 if channel == 0 else prediction*1.34
min8 = prediction*0.835 if channel == 0 else prediction*0.665
max8 = prediction*1.455 if channel == 0 else prediction*1.475
min9 = prediction*0.73 if channel == 0 else prediction*0.53
max9 = prediction*1.56 if channel == 0 else prediction*1.61
min99 = prediction*0.52 if channel == 0 else prediction*0.26
max99 = prediction*1.77 if channel == 0 else prediction*1.88
confi = st.selectbox('Select confidence ', options=['68.2%','86.6%','95%','99.7%'])
if confi == '68.2%':
    st.subheader('Predicted Sell-in price of {} is {} '.format(name, np.round(prediction*1.14 if channel == 0 else prediction*1.07,2)))
    st.write('Range of Sell-in price: {} - {}'.format(np.round(min7,2), np.round(max7,2)))
elif confi == '86.6%':
    st.subheader('Predicted Sell-in price of {} is {}'.format(name, np.round(prediction*1.14 if channel == 0 else prediction*1.07,2)))
    st.write('Range of Sell-in price: {} - {}'.format(np.round(min8,2), np.round(max8,2)))
elif confi == '95%':
    st.subheader('Predicted Sell-in price of {} is {}'.format(name, np.round(prediction*1.14 if channel == 0 else prediction*1.07,2)))
    st.write('Range of Sell-in price: {} - {}'.format(np.round(min9,2), np.round(max9,2)))
else:
    st.subheader('Predicted Sell-in price of {} is {}'.format(name, np.round(prediction*1.14 if channel == 0 else prediction*1.07,2)))
    st.write('Range of Sell-in price: {} - {}'.format(np.round(min99,2), np.round(max99,2))) 

