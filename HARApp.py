import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import accuracy_score
from utils import predict, convert_timestamp

root = tk.Tk()


class App:
    def __init__(self, master):
        self.linear_acc_path = ''
        self.gyroscope_path = ''
        self.master = master
        self.master.title("Human Activity Recognition App")

        self.label = tk.Label(master, text="Select input file:")
        self.label.pack()

        self.browse_linear_acc_button = tk.Button(master, text="Browse Linear Acceleration",
                                                  command=self.browse_linear_acc_csv)
        self.browse_linear_acc_button.pack()

        self.browse_gyroscope_button = tk.Button(master, text="Browse Gyroscope", command=self.browse_gyroscope_csv)
        self.browse_gyroscope_button.pack()

        self.run_button = tk.Button(master, text="Run", command=self.run_activity_recognition_model)
        self.run_button.pack()

        self.output_label = tk.Label(master, text="Output:")
        self.output_label.pack()

        self.output_text = tk.Text(master)
        self.output_text.pack()

    def browse_linear_acc_csv(self):
        """
        Push the button the browse linear acceleration csv file.
        """
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        self.linear_acc_path = file_path

    def browse_gyroscope_csv(self):
        """
        Push the button to browse gyroscope csv file.
        """
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        self.gyroscope_path = file_path

    def run_activity_recognition_model(self):
        """
        Push run, we will start the prediction.
        """
        # Linear acceleration file must be selected.
        if not hasattr(self, "linear_acc_path") or "Linear Accelerometer" not in self.linear_acc_path:
            self.output_text.insert(tk.END, "Please select an Linear Acceleration file first.")
            return

        # Gyroscope file must be selected.
        if not hasattr(self, "gyroscope_path") or "Gyroscope" not in self.gyroscope_path:
            self.output_text.insert(tk.END, "Please select an Gyroscope file first.")
            return

        # get the current directory of csv file
        curr_dir = os.path.dirname(os.path.abspath(self.linear_acc_path))

        # get the path of time.csv
        time_path = os.path.join(curr_dir, 'meta', 'time.csv')
        time_df = pd.read_csv(time_path)

        # get the start time
        start_time = time_df['system time'].loc[time_df['event'] == 'START'].values[0]

        linear_acc_df = pd.read_csv(self.linear_acc_path)
        gyroscope_df = pd.read_csv(self.gyroscope_path)

        predictions = predict(linear_acc_df, gyroscope_df)
        length = len(predictions)

        # get the true labels
        true_labels = [0 if 'walking' in curr_dir else 1] * length

        # get the accuracy
        accuracy = accuracy_score(true_labels, predictions)

        # get the datetime of each window
        times = linear_acc_df['Time (s)'].iloc[:length].values
        times = list(map(lambda x: convert_timestamp(start_time + x), times))

        outputs = list(map(lambda x: 'walking' if x == 0 else 'jumping', predictions))

        prediction_df = pd.DataFrame(zip(times, outputs), columns=['time', 'prediction'])

        output_file = 'outputs/' + str(start_time) + '_prediction.csv'
        prediction_df.to_csv(output_file, index=False)
        self.output_text.insert(tk.END, f"Outputs saved to {output_file}.\n")
        self.output_text.insert(tk.END, f"The accuracy is: {accuracy}.\n")

        fig, ax = plt.subplots(1, 2)
        ax[0].scatter(np.linspace(1, length, length), predictions)
        ax[0].set_title('results of prediction')
        ax[0].set_xlabel("Time Window")
        ax[0].set_ylabel("Activity")

        ax[1].scatter(np.linspace(1, length, length), true_labels)
        ax[1].set_title('results of true labels')
        ax[1].set_xlabel("Time Window")
        ax[1].set_ylabel("Activity")

        # Create a Matplotlib canvas widget to embed in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack()


app = App(root)
root.mainloop()
