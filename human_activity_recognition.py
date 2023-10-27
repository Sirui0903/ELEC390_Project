# ### Read the csv file

import os 
import numpy as np 
import pandas as pd 
from utils import walk_csv 

csv_path = 'csv'

# Get the paths of csv files 
csv_list = walk_csv(csv_path)

# Files in meta directory are not needed, so remove them. 
csv_list = list(filter(lambda x: 'meta' not in x, csv_list))
csv_list


# The linear accelerometer data and the groscope data are collected in different files,
# so we want to join them together. Besides, we need to label our data.
# 0 is for 'walking' and 1 is for 'jumping'. After that, we can store them as HDF5 file.

# First, we get the current directories of the csv files. 
curr_dirs = set(map(lambda x: os.path.join(x.split('\\')[0], x.split('\\')[1], x.split('\\')[2]), csv_list))

# Record the data each member 
member_record = {'member1': [], 'member2': [], 'member3': []}

for dir in curr_dirs:
    # Get the member name 
    member = dir.split("\\")[1]

    # Get gyroscope and linear_acceleration dataframe. 
    gyroscope, linear_acc, _ = os.listdir(dir)
    gyroscope_df, linear_acc_df = pd.read_csv(os.path.join(dir, gyroscope)), pd.read_csv(os.path.join(dir, linear_acc))

    # Join gyroscope and linear_acceleration together 
    df = pd.merge(left=gyroscope_df, right=linear_acc_df, on='Time (s)')

    # Label our data
    if 'walking' in dir: 
        df['label'] = 0
    else:
        df['label'] = 1

    # Add the raw data to member record     
    member_record[member].extend(df.to_numpy().tolist())

# Convert list to ndarray 
member_record = dict(map(lambda x: (x[0], np.array(x[1])), member_record.items()))
member_record

#                     ###  Step 2 Data storing###

import h5py
from sklearn.model_selection import train_test_split 
from utils import devide_signals 

# We first store the raw data in HDF5 file. 

# Create a HDF5 file
f = h5py.File("HAR.hdf5", "w")

# Create three groups under root '/'.
member1 = f.create_group("member1")
member2 = f.create_group("member2")
member3 = f.create_group("member3")

# Create a dataset under group "member1".
d1 = member1.create_dataset("data1", data=member_record['member1'])

# Create a dataset under group "member2".
d2 = member2.create_dataset("data2", data=member_record['member2'])

# Create a dataset under group "member3".
d3 = member3.create_dataset("data3", data=member_record['member3'])

# Then, we create a dataset. We should divide each signal into 5-second windows,
# shuffle the segmented data, and use 90% for training and 10% for testing.

# Create a group named dataset under root '/'.
dataset = f.create_group("dataset")

# Divide each signal into 5-second windows
signal_windows1 = devide_signals(member_record['member1'])
signal_windows2 = devide_signals(member_record['member2'])
signal_windows3 = devide_signals(member_record['member3'])

signal_windows = np.concatenate((signal_windows1, signal_windows2, signal_windows3))

# Split the dataset into training set and test set
train, test = train_test_split(signal_windows, test_size=0.1)
print("The size of training set: {}, the size of testing size: {}".format(train.shape, test.shape))

# Store the training set and test set in HDF5 file.  

# Create two groups under dataset.
training_set = dataset.create_group("train")
test_set = dataset.create_group("test")

# # Create a dataset under group "train".
train_data = training_set.create_dataset("train_data", data=train)

# Create a dataset under group "test".
test_data = test_set.create_dataset("test_data", data=test)

# Save and exit the file.
f.close()

#                ### Step 3 Visualization###
# First of all, we will plot the linear acceleration vs. time. 

from matplotlib import pyplot as plt 
from utils import liear_acc_plot, linear_acc_freq_plot

# Read the linear acceleration from csv files
linear_acc_csv_list = list(filter(lambda x : 'Linear' in x, csv_list))

for linear_acc_csv in linear_acc_csv_list:
    print("The file is: {}".format(linear_acc_csv))
    df = pd.read_csv(linear_acc_csv)
    liear_acc_plot(df)

# According to the above figures, we can see that when we are walking,
# the distribution of three axises is pretty close regardless of the position of our phone.
# On the other hand, when we are junmping, one of the axises will have the largest amplitude,
# while the two other axies remain close distribution. For example, when we are jumping with the phone carried in hand,
# the linear acceleration in X and Y axis is pretty close and amplitude is not larger than 10(m/s^2).
# However, the linear acceleration in Z axises have the largest amplitude which is close to 40(m/s^2).

# Then we want to plot linear acceleration vs. frequency.

for linear_acc_csv in linear_acc_csv_list:
    print("The file is: {}".format(linear_acc_csv))
    df = pd.read_csv(linear_acc_csv)
    linear_acc_freq_plot(df)

# According to the above figures linear acceleration vs. freqency has the same result as we observed when we plot acceleration vs. time.

#                 ### Step 4 Pre-processing ###

from utils import denoise, ava_filter, complementary_filter, sensor_fusion 

# #### Noise Reducing

# We first use a sample data to check how the filter works

linear_acc_sample = pd.read_csv('csv/member1/jumping-carry in hand-2min 2023-03-14 20-25-56/Linear Accelerometer.csv')

# Plot the data before denoising
liear_acc_plot(linear_acc_sample)

# Apply average_filter
X, Y, Z = linear_acc_sample['X (m/s^2)'].to_list(), linear_acc_sample['Y (m/s^2)'].to_list(), linear_acc_sample['Z (m/s^2)'].to_list() 
linear_acc_sample['X (m/s^2)'], linear_acc_sample['Y (m/s^2)'], linear_acc_sample['Z (m/s^2)'] = ava_filter(X), ava_filter(Y), ava_filter(Z)
# Plot the data after denoising
liear_acc_plot(linear_acc_sample)

X_train, y_train = train[:, :, 1: -1], train[:, -1, -1]
X_test, y_test = test[:, :, 1: -1], test[:, -1, -1]

X_train, X_test = denoise(X_train), denoise(X_test)
print("The size of training set: {}, the size of testing size: {}".format(X_train.shape, X_test.shape))

# #### Linear Acceleration Adjustment

# In some cases, it may be necessary to adjust the acceleration data to account for the orientation of the sensor relative to the body of the person performing the activity. This is because the sensor may be positioned in a way that is not aligned with the axes of the body. For example, if the sensor is attached to the wrist, the acceleration readings in the x, y, and z directions may not correspond to the movements of the arm.  
# 
# In such cases, it may be necessary to apply a transformation to the acceleration data to account for the orientation of the sensor. This transformation is typically based on the known orientation of the sensor relative to the body, which can be obtained through calibration.  
#  
# Another reason to adjust the acceleration data is to account for the effects of gravity. Since the acceleration due to gravity is constant and acts in the same direction regardless of the orientation of the sensor, it can be subtracted from the raw acceleration data to obtain the acceleration due to the movement of the body.  
#  
# Now we have collected linear acceleration and gyroscope data, we can use the gyroscope data to adjust the acceleration data by compensating for the orientation of the sensor. This is known as sensor fusion and can be done using a technique called complementary filtering.  
#  
# Complementary filtering is a technique that combines the measurements from multiple sensors, such as an accelerometer and a gyroscope, to obtain a more accurate estimate of the orientation of the sensor. The idea behind complementary filtering is to use the gyroscope data to estimate the short-term changes in the orientation of the sensor, and use the accelerometer data to estimate the long-term changes in the orientation of the sensor.  
#  
# To apply complementary filtering to adjust the acceleration data based on the gyroscope data, we can follow these steps:
#  
# 1. Integrate the gyroscope data to obtain the estimated change in orientation of the sensor. This can be done by applying the trapezoidal rule to the gyroscope readings to obtain the angular velocity of the sensor, and then integrating this angular velocity over time to obtain the estimated change in orientation.
#  
# 2. Convert the estimated change in orientation from step 1 to a rotation matrix that represents the change in orientation of the sensor.
#  
# 3. Use the rotation matrix from step 2 to adjust the acceleration data. This can be done by multiplying the acceleration data by the rotation matrix to obtain the adjusted acceleration data.

# Still, we first use the sample data to check how this works

gyroscope_sample = pd.read_csv('csv/member1/jumping-carry in hand-2min 2023-03-14 20-25-56/Gyroscope.csv')

print('Plot the sample data before sensor fusion')
liear_acc_plot(linear_acc_sample)

linear_acceleration = linear_acc_sample[['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)']].to_numpy()
gyroscope = gyroscope_sample[["X (rad/s)","Y (rad/s)","Z (rad/s)"]].to_numpy()

linear_acceleration_adjusted = complementary_filter(linear_acceleration, gyroscope)
linear_acc_sample[['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)']] = linear_acceleration_adjusted

print('Plot the sample data after sensor fusion') 
liear_acc_plot(linear_acc_sample)

# get linear acceleration and gyroscope
linear_acc_train, gyroscope_train = X_train[:, :, :3], X_train[:, :, 3:]
linear_acc_test, gyroscope_test = X_test[:, :, :3], X_test[:, :, 3:]

# Apply sensor fusion
linear_acc_train, linear_acc_test = sensor_fusion(linear_acc_train, gyroscope_train), sensor_fusion(linear_acc_test, gyroscope_test)

print(linear_acc_train.shape, linear_acc_test.shape)

#                   ### Step 5 Feature Extraction ###


from utils import feature_extraction, remove_outliers


linear_acc_train, linear_acc_test = feature_extraction(linear_acc_train), feature_extraction(linear_acc_test)
print(linear_acc_train.shape, linear_acc_test.shape)

# After feature extraction (next step), we detect and remove the outliers in linear acceleration data.

linear_acc_train, train_indices = remove_outliers(linear_acc_train)
linear_acc_test, test_indices = remove_outliers(linear_acc_test)
print(linear_acc_train.shape, linear_acc_test.shape)


#                   ### Step 6 Creating a classifier ###

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from utils import normalization, plot_learning_curve

# normalize the data
linear_acc_train, linear_acc_test = normalization(linear_acc_train), normalization(linear_acc_test)

# delete the outliers' label
y_train, y_test = y_train[train_indices], y_test[test_indices]

# Train logistic regression model
model = LogisticRegression()
model.fit(linear_acc_train, y_train)

# Evaluate model on testing set
y_pred = model.predict(linear_acc_train)

train_accuracy = accuracy_score(y_train, y_pred)
train_precision = precision_score(y_train, y_pred, average='weighted')
train_recall = recall_score(y_train, y_pred, average='weighted')
train_f1 = f1_score(y_train, y_pred, average='weighted')

print("Train Accuracy:", train_accuracy)
print("Train Precision:", train_precision)
print("Train Recall:", train_recall)
print("Train F1-score:", train_f1)
print(classification_report(y_train, y_pred))

# Evaluate model on testing set
y_pred = model.predict(linear_acc_test)

test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, average='weighted')
test_recall = recall_score(y_test, y_pred, average='weighted')
test_f1 = f1_score(y_test, y_pred, average='weighted')

print("Test Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1-score:", test_f1)
print(classification_report(y_test, y_pred))

train_sizes, train_scores, test_scores = learning_curve(model, linear_acc_train, y_train, cv=5)
plot_learning_curve(train_sizes, train_scores, test_scores)

import joblib

# save the model
joblib.dump(model, 'model_saved/logistic_regression.m')


