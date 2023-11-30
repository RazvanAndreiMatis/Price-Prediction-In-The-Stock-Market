# Copyright(C) 2023 Razvan Andrei Matis Girda

# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


# Import the libraries
import math
import numpy as np  # Library used to create arrays and reshape the data
import matplotlib.pyplot as plt  # Library used to create two dimension graphs from the data arrays
import datetime as dt  # Library used to obtain the range of the data used to train the model
import yfinance as yf  # Library used to obtain the market data used to train the model
import time

from sklearn.preprocessing import MinMaxScaler  # Transform features by scaling each feature to a given range
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
from pandas_datareader import data as pdr  # Used to access financial data and is imported as a DataFrame

f = open('file.py', 'w')

# Load Data

# Company used to predict the stock market price
company = 'GOOG'

# Start date
start = dt.datetime(2008, 1, 18)

# End date
end = dt.datetime(2023, 1, 18)

# Obtain the data from yahoo finance
yf.pdr_override()
df = pdr.get_data_yahoo(company, start, end)

# Visualize the closing price for the interval of time established
plt.figure(figsize=(18, 9))
plt.title(f'{company} Close Price History', fontsize=18)
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.savefig(f'C:/Users/matis/OneDrive/Escritorio/Nueva_carpeta_2/TFG/TFG/Resultados/A/{company}_history.png')


# Create a new dataframe with only the 'Close' column
data = df.filter(['Close'])

# Covert the dataframe to a numpy array
dataset = data.values

# Scale the data
# This is always done to the input data before presenting the data to the nural network
# It is scaled to values between 0 and 1 (it could be 0, 1 or anything in between)
scaler = MinMaxScaler(feature_range=(0, 1))
# Computes the minimum and maximum values to be used for scaling and transforms the data based on these two values
scaled_data = scaler.fit_transform(dataset)

# Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)
# Create the training data set
# Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]

prediction_days = 60

# Split the data into x and y data sets
# Independent variables
x = []
# Dependent variables
y = []

for i in range(prediction_days, len(train_data)):
    # Contains 60 values, from position 0 to position 59
    x.append(train_data[i - prediction_days:i, 0])
    # Contains 1 value, at position 60. The one that we want our model to predict
    y.append(train_data[i, 0])

# Convert the x and y to numpy arrays to train the LSTM model
x, y = np.array(x), np.array(y)

# Reshape the data
# The LSTM network expects the input to be 3 dimensional in the form of
# number of samples, number of time steps and number of features
# Now the dataset is 2 dimensional

# The number of samples is equal to the number of rows
# The number of time steps is equal to the number of columns
# The number of features is equal to 1 (the closing price)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Build the LSTM model
model = Sequential()
# 1st parameter is the number of neurons
# 2nd parameter
#   If true means another LSTM layer will be used
#   If false means another LSTM layer will not be used
# 3rd parameter, because this is the first layer we need to give it an input shape
model.add(LSTM(100, return_sequences=True, input_shape=(x.shape[1], 1)))
model.add(LSTM(100))
model.add(Dense(30))
model.add(Dense(1))  # Prediction of the next closing price value

# Compile the model
# The optimizer is used to improve upon the loss function
# The loss function is used to measure how well the model did on training
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Set a timer to know the model's total training time
# Start the timer
start_timer = time.time()

# Train the model
# fit is another name for train
# Epochs refers to the number of times the model's going to see the data
# Batch size refers to the number of units the model's going so see every time
model.fit(x, y, batch_size=32, epochs=50)

# End the timer
end_timer = time.time()
total_training_time = end_timer - start_timer

# Create the test data set
# Create a new array containing scaled values
test_data = scaled_data[training_data_len - prediction_days:, :]

# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]  # Values that we want the model to predict

for i in range(prediction_days, len(test_data)):
    x_test.append(test_data[i - prediction_days:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values
predictions = model.predict(x_test)
# Unscaling the values
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
# Lower values indicate a better fit
mse = mean_squared_error(y_test, predictions)
mse_rounded = round(mse, 5)

rmse = np.sqrt(mse)
rmse_rounded = round(rmse, 5)

# Get the coefficient of determination (R2)
r2 = r2_score(y_test, predictions)
r2_rounded = round(r2, 5)

f.write('MSE = ' + repr(mse_rounded) + '\n')
f.write('RMSE = ' + repr(rmse_rounded) + '\n')
f.write('R2 = ' + repr(r2_rounded) + '\n')

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions


# Visualize the data (Dispersion & Linear Regression Graph)
plt.figure(figsize=(18, 9))
plt.title(f'{company} Dispersion & Linear Regression Graph', fontsize=18)
plt.xlabel('Real data', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.scatter(y_test, predictions, label=f'$R^2={r2:.2f}$')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', label='Regression Line')
plt.legend(loc='lower right')
plt.savefig(f'C:/Users/matis/OneDrive/Escritorio/Nueva_carpeta_2/TFG/TFG/Resultados/A/{company}_prediction_r2.png')



# Visualize the data (Datasets)
plt.figure(figsize=(18, 9))
plt.title(f'{company} Close Price History (Datasets)', fontsize=18)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close']])
plt.legend(['Training values', 'Test values'], loc='lower right')
plt.savefig(f'C:/Users/matis/OneDrive/Escritorio/Nueva_carpeta_2/TFG/TFG/Resultados/A/{company}_history_test.png')



# Visualize the data
plt.figure(figsize=(18, 9))
plt.title(f'{company} Close Price Prediction', fontsize=18)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
legend2 = plt.legend(['Training values', 'Test values', 'Predicted values'], loc='lower right')
# Wait for the legend to be fully rendered
plt.draw()
plt.pause(0.3)
# Get the axis coordinates for the legend
ax2 = plt.gca()
legend2_bbox = legend2.get_window_extent().transformed(ax2.transAxes.inverted())
# Calculate coordinates for the text above the legend
text2_x = legend2_bbox.x0 + (legend2_bbox.x1 - legend2_bbox.x0) / 2
text2_y = legend2_bbox.y1 + 0.01
# Display the R2, MSE and RMSE
plt.text(text2_x, text2_y, f'R2 = {r2_rounded}', transform=ax2.transAxes, ha='center', va='bottom', fontsize=10)
plt.text(text2_x, text2_y + 0.03, f'MSE = {mse_rounded}', transform=ax2.transAxes, ha='center', va='bottom', fontsize=10)
plt.text(text2_x, text2_y + 0.06, f'RMSE = {rmse_rounded}', transform=ax2.transAxes, ha='center', va='bottom', fontsize=10)
plt.savefig(f'C:/Users/matis/OneDrive/Escritorio/Nueva_carpeta_2/TFG/TFG/Resultados/A/{company}_prediction.png')



# Visualize the data (zoom)
plt.figure(figsize=(18, 9))
plt.title(f'{company} Close Price Prediction (Zoomed)', fontsize=18)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(valid[['Close', 'Predictions']])
legend3 = plt.legend(['Test values', 'Predicted values'], loc='lower right')
# Wait for the legend to be fully rendered
plt.draw()
plt.pause(0.3)
# Get the axis coordinates for the legend
ax3 = plt.gca()
legend3_bbox = legend3.get_window_extent().transformed(ax3.transAxes.inverted())
# Calculate coordinates for the text above the legend
text3_x = legend3_bbox.x0 + (legend3_bbox.x1 - legend3_bbox.x0) / 2
text3_y = legend3_bbox.y1 + 0.01
# Display the R2, MSE and RMSE
plt.text(text3_x, text3_y, f'R2 = {r2_rounded}', transform=ax3.transAxes, ha='center', va='bottom', fontsize=10)
plt.text(text3_x, text3_y + 0.03, f'MSE = {mse_rounded}', transform=ax3.transAxes, ha='center', va='bottom', fontsize=10)
plt.text(text3_x, text3_y + 0.06, f'RMSE = {rmse_rounded}', transform=ax3.transAxes, ha='center', va='bottom', fontsize=10)
plt.savefig(f'C:/Users/matis/OneDrive/Escritorio/Nueva_carpeta_2/TFG/TFG/Resultados/A/{company}_prediction_zoomed.png')



# Predict next day
df = pdr.get_data_yahoo(company, start, end)
# Create new dataframe
new_df = df.filter(['Close'])
# Get the last 60 days closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
# Scale the data to values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
# Create an empty list
# Append past 60 days
X_test = [last_60_days_scaled]
# Convert the data set to a numpy array
X_test = np.array(X_test)
# Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Get the predicted scaled price
predicted_price = model.predict(X_test)
# Undo the scaling
predicted_price = scaler.inverse_transform(predicted_price)

f.write('Predicted Price = ' + repr(predicted_price) + '\n')

# Get the actual price for the day predicted
# Start date
start = dt.datetime(2023, 1, 18)
# End date
end = dt.datetime(2023, 1, 19)

df2 = pdr.get_data_yahoo(company, start, end)

f.write('Actual price = ' + repr(df2['Close']) + '\n')

f.write('Total training time = ' + repr(total_training_time) + '\n')

f.close()
