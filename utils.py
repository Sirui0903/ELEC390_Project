import os
import numpy as np
import datetime
from matplotlib import pyplot as plt
from scipy.fftpack import fft
from scipy.signal import find_peaks 
import joblib

model = joblib.load('model_saved/logistic_regression.m')

# The Sensor rate is 10, which means we collect the data 10 times per second.
# Since a window contains 5s-data, we set the window size to 50.
window_size = 50


def walk_csv(path):
    """
    Walk through the csv files under a certain path.
    """
    csv_list = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                csv_list.append(os.path.join(root, file))

    return csv_list


def devide_signals(array):
    """
    Divide each signal into 5-second windows.
    """
    signal_windows = []
    for i in range(0, len(array) - window_size + 1):
        signal_windows.append(array[i: i + window_size].tolist())
    return np.array(signal_windows)


def liear_acc_plot(df):
    """
    Plot the linear acceleration per second.
    """
    plt.plot(df['X (m/s^2)'].loc[::10], label='X (m/s^2)')
    plt.plot(df['Y (m/s^2)'].loc[::10], label='Y (m/s^2)')
    plt.plot(df['Z (m/s^2)'].loc[::10], label='Z (m/s^2)')
    plt.legend()
    plt.xlabel('Time(s)')
    plt.ylabel('Linear Acceleration(m/s^2)')
    plt.show()


def linear_acc_freq_plot(df):
    """
    Apply fast fourier transformation to linear acceleration and plot linear acceleration vs. frequency.
    """
    linear_acc_X = df['X (m/s^2)'].to_numpy()
    linear_acc_Y = df['Y (m/s^2)'].to_numpy()
    linear_acc_Z = df['Z (m/s^2)'].to_numpy()

    # Apply fft to linear acceleration
    linear_acc_X_freq = fft(linear_acc_X)
    linear_acc_Y_freq = fft(linear_acc_Y)
    linear_acc_Z_freq = fft(linear_acc_Z)

    plt.plot(linear_acc_X_freq, label='X (Hz)')
    plt.plot(linear_acc_Y_freq, label='Y (Hz)')
    plt.plot(linear_acc_Z_freq, label='Z (Hz)')
    plt.legend()
    plt.xlabel('Frequency(Hz)')
    plt.show()


def ava_filter(x, filt_length=10):
    """
    Build a average filter for denoising.
    """
    N = len(x)
    res = []
    for i in range(N):
        if i <= filt_length // 2 or i >= N - (filt_length // 2):
            temp = x[i]
        else:
            sum = 0
            for j in range(filt_length):
                sum += x[i - filt_length // 2 + j]
            temp = sum * 1.0 / filt_length
        res.append(temp)
    return res


def denoise(data):
    denoise = []
    for d in data:
        denoise.append(np.array(ava_filter(d)))
    return np.array(denoise)


def complementary_filter(linear_acc_raw, gyro_raw):
    def get_gyro_scale():
        """
        'gyro_scale' is a scaling factor that is used to adjust the integration of the gyroscope data in the complementary filtering algorithm.
        The value of 'gyro_scale' should be chosen based on the specific gyroscope used in your sensor, and it depends on factors such as the sensitivity and noise of the gyroscope.

        The purpose of 'gyro_scale' is to balance the gyroscope data and the accelerometer data in the complementary filtering algorithm.
        The gyroscope data provides short-term orientation changes that are more accurate, but can drift over time.
        The accelerometer data provides long-term orientation changes that are less accurate, but are more reliable over time.
        By adjusting the integration of the gyroscope data using 'gyro_scale', you can balance these two sources of orientation information to obtain a more accurate estimate of the orientation of the sensor.

        To determine the appropriate value of 'gyro_scale' for your specific sensor, you can perform some calibration experiments to measure the bias and noise of the gyroscope.
        One way to do this is to place the sensor on a stable surface and record the gyroscope readings over a period of time. By averaging the gyroscope readings, you can estimate the bias of the gyroscope.
        By calculating the standard deviation of the gyroscope readings, you can estimate the noise of the gyroscope.

        Once you have measured the bias and noise of the gyroscope, you can use these values to determine the appropriate value of 'gyro_scale'.
        One common approach is to set 'gyro_scale' to the ratio of the expected noise of the accelerometer data to the expected noise of the gyroscope data.
        This ensures that the integration of the gyroscope data is adjusted to balance the noise of the two sensors.
        """
        # Measure bias and noise of gyroscope and accelerometer
        gyro_bias = np.mean(gyro_raw, axis=0)
        gyro_noise = np.std(gyro_raw, axis=0)
        accel_noise = np.std(linear_acc_raw, axis=0)

        # Calculate gyro_scale
        gyro_scale = accel_noise / gyro_noise
        return gyro_scale

        # Gyroscope scaling factor

    gyro_scale = get_gyro_scale()

    # Initialize complementary filter variables
    orientation = np.eye(3)
    gyro_integrated = np.zeros((3,))

    accel_adjusted = np.array([])
    # Complementary filtering loop
    for i in range(len(linear_acc_raw)):
        # Integrate gyroscope data
        gyro_integrated += gyro_raw[i] * gyro_scale
        # Calculate rotation matrix from gyroscope data
        delta_rotation = np.eye(3) + np.cross(gyro_integrated, orientation, axisa=0, axisb=0)
        # Normalize rotation matrix to ensure orthonormality
        orientation = np.dot(delta_rotation, orientation)
        orientation /= np.linalg.norm(orientation, axis=1, keepdims=True)
        # Adjust acceleration data using rotation matrix
        accel_adjusted = np.dot(orientation, linear_acc_raw.T).T
    return accel_adjusted


def sensor_fusion(linear_acc, gyro):
    """
    Apply complementary filtering to every batch of linear acceleration. 
    """
    length = len(linear_acc)

    linear_acc_adjusted = []
    for i in range(length):
        linear_acc_adjusted.append(complementary_filter(linear_acc[i], gyro[i]))

    return np.array(linear_acc_adjusted)


def rms(data):
    """
    The RMS acceleration in each axis provides a measure of the overall intensity of the acceleration in each direction. 
    It is often used as a feature for human activity recognition, as it can capture differences in the intensity of movement between different activities.
    """
    return np.sqrt(np.mean(np.square(data)))


def correlation(array):
    """
    Correlation is a measure of the linear relationship between the acceleration signals in each axis.
    This feature can provide information about the direction and orientation of movement.
    """
    return np.corrcoef(array.T).flatten() 


def skewness(data):
    """
    Skewness is a measure of the asymmetry of a distribution. 
    It indicates whether the data is skewed to the left or right. 
    """
    return  np.mean((data - np.mean(data))**3) / np.mean((data - np.mean(data))**2)**1.5


def kurtosis(data):
    """
    Kurtosis, on the other hand, is a measure of the peakedness or flatness of a distribution. 
    It indicates how much of the variance of the data is due to extreme values (outliers).
    """
    return np.mean((data - np.mean(data))**4) / np.mean((data - np.mean(data))**2)**2


def energy(data):
    """
    Energy is a measure of the total power of a signal. 
    It is often used to quantify the intensity of a signal or the amount of work done by a system. 
    """
    return np.sum(data**2)


def dominant_frequency(data):
    """
    Dominant frequency is the frequency that has the highest magnitude in a signal.
    It is often used to identify the primary frequency component in a signal. 
    """
    fft = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data))
    idx = np.argmax(np.abs(fft))
    dominant_freq = freqs[idx]
    return dominant_freq 


def zero_crossing(data):
    """
    Zero-crossing is a measure of the number of times a signal crosses the zero axis. 
    It is often used to quantify the number of times a signal changes direction or to detect sudden changes in the signal. 
    """
    signs = np.sign(data)
    return np.sum(np.abs(np.diff(signs))) / 2


def peaks_num(data):
    """
     A peak in the acceleration signal indicates a change in the direction of the acceleration, which could be caused by a change in movement or activity.
     By counting the number of peaks in each axis, we can get a sense of how much the subject is moving or changing direction in each axis. 
    """
    _, peaks = find_peaks(data)
    return len(peaks)


def wave_length(data):
    """
    Waveform length is a measure of the cumulative length of a signal's waveform.
    It is often used to quantify the complexity of a signal or the amount of variation in a time series.
    """
    return np.sum(np.abs(np.diff(data)))


def feature_extraction(data):
    """
    We will extract the following features:
    1. Mean acceleration in each axis
    2. Standard deviation of acceleration in each axis
    3. Median acceleration in each axis
    4. Maximum acceleration in each axis
    5. Minimum acceleration in each axis
    6. Range of acceleration in each axis
    7. Root Mean Square (RMS) acceleration in each axis
    8. Skewness of acceleration in each axis
    9. Kurtosis of acceleration in each axis
    10. Correlation between acceleration in different axes (e.g., correlation between x and y acceleration)
    11. Energy of acceleration in each axis (sum of squared values)
    12. Dominant frequency of acceleration in each axis (calculated using Fourier transform)
    13. Zero-crossing rate of acceleration in each axis (number of times the acceleration signal crosses zero)
    14. Number of peaks in acceleration signal in each axis (using a peak detection algorithm)
    15. Waveform length of acceleration in each axis (sum of the absolute differences between adjacent samples)
    """

    feature_functions = [np.mean, np.std, np.max, np.min, np.median, np.ptp, rms, skewness, kurtosis, energy, dominant_frequency, zero_crossing, peaks_num, wave_length]
    
    data_featured = []
    for d in data:
        features = []
        for i in range(3):
            for func in feature_functions:
                features.append(func(d[:, i]))
        features.extend(correlation(d))
        data_featured.append(features)
    return np.array(data_featured)


def remove_outliers(data, threshold=3):
    """
    Compute the z-score for each data point in the dataset, which is the number of standard deviations away from the
    mean. Data points with a z-score above a certain threshold are considered outliers and can be removed from the
    dataset. For example, a threshold of 3 would identify data points with a z-score greater than 3 standard
    deviations from the mean.
    """
    # compute z-scores for each feature

    zscores = np.abs(normalization(data))

    # compute maximum z-score for each data point
    max_zscores = np.max(zscores, axis=1)

    # identify data points with z-score above threshold
    outlier_indices = np.where(max_zscores > threshold)

    # remove outliers from dataset
    cleaned_data = np.delete(data, outlier_indices[0], axis=0)

    # record the original indices
    original_indices = np.delete(np.arange(len(data)), outlier_indices[0])

    return cleaned_data, original_indices


def normalization(data):
    """
    Normalize the data so that it becomes suitable for logistic regression
    """
    mean, std = np.mean(data, axis=0), np.std(data, axis=0)
    std = np.array(list(map(lambda x: x if x != 0 else np.ptp(std), std)))
    return (data - mean) / std


def convert_timestamp(timestamp):
    """
    Convert timestamp to datetime.
    """
    d = datetime.datetime.fromtimestamp(timestamp / 1000)
    time_str = d.strftime("%Y-%m-%d %H:%M:%S.%f")
    return time_str


def predict(linear_acc_df, gyroscope_df):
    """
    Predict the label of each widow given linear acceleration and gyroscope
    """
    linear_acc_array = linear_acc_df.iloc[:, 1:].to_numpy()
    gyroscope_array = gyroscope_df.iloc[:, 1:].to_numpy()

    linear_acc_windows, gyroscope_windows = devide_signals(linear_acc_array), devide_signals(gyroscope_array)
    linear_acc_windows, gyroscope_windows = denoise(linear_acc_windows), denoise(gyroscope_windows)
    linear_acc_windows = sensor_fusion(linear_acc_windows, gyroscope_windows)
    inputs = feature_extraction(linear_acc_windows)

    outputs = model.predict(inputs)
    return outputs


def plot_learning_curve(train_sizes, train_scores, test_scores):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Logistic Regression Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()