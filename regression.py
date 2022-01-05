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
data = pd.read_csv('final_data_.csv')

# Data preprocessing
data = data.reset_index()
data = data.dropna()
data = data.drop_duplicates()
data_new = data[['channel','1m', 'weight', 'retail', 'selling']]

x = data_new[['retail', '1m','weight', 'channel']]
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
st.title("Data Science")
st.write('## Price Regression Project')

menu = ['Overview', 'Build Project', 'New Prediction']
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Overview':
    st.subheader('Overview')
    st.write("""
    #### The data has been split into two groups:
    - training set (train.csv):
    The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.
    - test set (test.csv):
    The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.
    - gender_submission.csv:  a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.
    """)
elif choice == 'Build Project':
    st.subheader('Build Project')    
    st.write('#### Data Preprocessing')    
    st.write('#### Show data:')
    st.table(data_new.head(5))

    st.write('#### Build model and evaluation')      
    st.write('Mean squared error: {}'.format(round(mse,2)))  
    st.write('Root mean squared error: {}'.format(round(rmse,2)))  
    st.write('Mean absolute error: {}'.format(round(mae,2)))  
    st.table(predict_df.head(10))      
    

    st.write('#### Visualization')
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    
    visualizer = PredictionError(model)
    visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(x_test, y_test)  # Evaluate the model on the test data
    visualizer.ax.set_title("Prediction Error for Gradient Boosting Regressor", fontsize= 20, fontweight = 'bold')

    visualizer.ax.set_xlabel("Actual selling", fontsize= 18)
    visualizer.ax.set_ylabel("Predicted selling", fontsize= 18)
    visualizer.finalize()
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='r', lw=3, scalex=False, scaley=False)
    plt.show()
    
    st_yellowbrick(visualizer)     

elif choice == 'New Prediction':
    st.subheader('Make new prediction')
    st.write('#### Input/Select data')
    name = st.text_input('Your name')
    
    channel = st.selectbox('Channel (0: Non-DI  1: DI)', options=[0, 1])
    retail = st.slider('Retail ($)', 0.2,600.0,0.01)
    weight = st.slider('Weight (lbs)', 0.0, 150.0, 0.01)
    dim = st.slider('Dimension (inch3)', 0.0, 16000.0, 1.0)

    # make new prediction
    # channel = 0 if channel == 'Non-DI' else 1
    new_data_1 = [[retail, dim, weight, channel]]
    prediction = model.predict(new_data_1)
    st.subheader('Selling prediction of {} is {}'.format(name, prediction))  
