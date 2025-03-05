#Importing libraries
import streamlit as st
import pandas as pd
import numpy as np

#Importing models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#Gathering the dataset
df = pd.read_csv('Energy_Consumption_Model\energy_consumption_data.csv')

#Selecting target and features
x = df.drop('Energy_Consumption', axis=1)
y = df['Energy_Consumption']

#Dividing the dataset to train & test
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=23)

#Building the models
lr = LinearRegression()
ridge = Ridge(alpha=0.1)
tree = DecisionTreeRegressor(max_depth=5)
rfr = RandomForestRegressor(n_estimators=10)

#Training the models
lr.fit(x_train,y_train)
ridge.fit(x_train,y_train)
tree.fit(x_train,y_train)
rfr.fit(x_train,y_train)


lr_score = lr.score(x_test,y_test)
ridge_score = ridge.score(x_test,y_test)
tree_score = tree.score(x_test,y_test)
rfr_score = rfr.score(x_test,y_test)


#Building streamlit
st.title('Energy Consumption Prediction App')
st.write('This app predicts energy consumption based on multiple factors like temperature, humidity, building type, and holidays.')


st.markdown('Feature Explanations:')
st.markdown('Temperature : The ambient temperature in degrees Celsius.')
st.markdown('Humidity (%) : The amount of moisture in the air.')
st.markdown('Wind Speed (km/h) : The speed of wind in kilometers per hour.')
st.markdown('Building Type : Different type of buildings have different energy consumption patterns.')
st.markdown('Number of Employees :The count of the employees in the building')
st.markdown('Is it a Holiday? : 0 for no and 1 for yes.')

model_option = st.selectbox('Choose a Prediction Model:', ['Linear Regression', 'Ridge Regression (α=0.1)','Decision Tree Regressor','Random Forest Regressor'])

st.sidebar.header('Enter the features')
day = st.sidebar.slider('Day of the year',1,365,100)
temperature = st.sidebar.slider('Temperature (°C)',-50,50,20)
humidity = st.sidebar.slider('Humidity (%)',0,100,50)
wind_speed = st.sidebar.slider('Wind Speed (km/h)',0,100,10)
building_type = st.sidebar.slider('Building type',0,2,1)
employee_count = st.sidebar.slider('Employee Count',0,5000,100)
holiday = st.sidebar.slider('Holiday',0,1,0)

input_data = np.array([day,temperature,humidity,wind_speed,building_type,employee_count,holiday]).reshape(1,-1)

if model_option == "Linear Regression":
    prediction = lr.predict(input_data)[0]
    score = lr_score

elif model_option == 'Ridge Regression (α=0.1)':
    prediction = ridge.predict(input_data)[0]
    score = ridge_score

elif model_option == 'Decision Tree Regressor':
    prediction = tree.predict(input_data)[0]
    score = tree_score

elif model_option == 'Random Forest Regressor':
    prediction = rfr.predict(input_data)[0]
    score = rfr_score

else:
    prediction = 0
    score = 0

st.subheader('Predicted Energy Consumption (kWh)')
st.write(f'**{prediction:.2f} kWh**')

st.subheader("📊 Model Performance (R² Score) (0-1)")
st.write(f'**{score:.3f}**')
