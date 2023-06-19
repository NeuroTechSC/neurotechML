import threading
import time
import json
import logging
import pandas as pd
from queue import Queue
from collections import deque
from tensorflow.keras.models import load_model

class ThreadClass(threading.Thread):
    def __init__(self, threadID, csv_file, model_path, button_event, window_size=5, memory_limit=1e6):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.daemon = True  # The thread will exit when the main program exits
        self.csv_file = csv_file  # Mock CSV file path
        self.model = load_model(self.model_path)  # LSTM/RNN model
        self.memory_limit = memory_limit  # Memory limit for the DataFrame
        self.button_event = button_event # Event to trigger phoneme prediction
        self.emg_data = pd.DataFrame()  # DataFrame to store EMG data
        self.predicted_phonemes = []  # List to store predicted phonemes
        self.lock = threading.Lock()  # Lock for thread-safe operations
        self.predict_queue = Queue()  # Queue to store predictions
        self.window_size = window_size  # Number of data points to be used for prediction
        self.phoneme_id = 0 # ID for each phoneme prediction

        # Logging
        
        # Initialize the logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(ch)
        
        self.logger.info("Thread initialized.")

    def run(self):
        self.logger.info("Thread started.")

        # Start a queue to hold timestamps for each data point
        timestamp_queue = deque()

        # Read the first 4 rows of the CSV file to get rid of the garbage data
        garbage = pd.read_csv(self.csv_file, nrows=4)

        while True:
            # Assume each line of CSV represents a 4ms timestep of data, read only the desired channels from the csv file
            data_chunk = pd.read_csv(self.csv_file, nrows=self.window_size, usecols=['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3'])

            # Rename the channels
            data_chunk.columns = ['DLI', 'OOS', 'OOI', 'PLA']

            # Append the data to the DataFrame
            self.emg_data = pd.concat([self.emg_data, data_chunk])
            
            # Record the current timestamp in the queue
            timestamp_queue.append(time.time())

            if self.button_event.is_set():
                self.logger.info("Button is in 'running' state.")

                # If the button is in the 'running' state
                self.predict_phoneme()
                
                # If DataFrame memory usage exceeds limit, drop the oldest data
                if self.emg_data.memory_usage(index=True, deep=True).sum() > self.memory_limit:
                    self.emg_data = self.emg_data.iloc[self.window_size:]
            else:
                self.logger.info("Button is in 'stopped' state.")

                # If the button is in the 'stopped' state
                self.process_phonemes()
            
            self.logger.info("Sleeping for 0.004 seconds...")
            
            # Pause before next iteration
            time.sleep(0.004)  # 4ms pause to simulate real-time data streaming

    def predict_phoneme(self):
        self.logger.info("Predicting phoneme...")

        # Get the most recent data for prediction
        recent_data = self.emg_data.tail(X)  # X is the number of time steps for each phoneme prediction
        
        # Preprocess the data
        input_data = self.preprocess(recent_data)

        # Predict phoneme
        phoneme = self.model.predict(input_data)
        
        # Store the predicted phoneme and add it to the queue for the main application to retrieve
        self.predicted_phonemes.append(phoneme)
        
        # Create a response to be sent to the main application
        response = {
        "responseType": "singlePhoneme",
        "body": {
            "class": phoneme,
            "id": self.phoneme_id,
            "row": recent_data.shape[0],
            "col": recent_data.shape[1],
            "data": recent_data.values.tolist()
            }
        }

        # Add the response to the queue
        self.predict_queue.put(json.dumps(response))
        
        # Increment the phoneme ID
        self.phoneme_id += 1

    def preprocess(self, data):
        # Data preprocessing method to be implemented
        return data

    def process_phonemes(self):
        self.logger.info("Processing phonemes...")

        # Initialize the processed list with the first phoneme
        processed_phonemes = [self.predicted_phonemes[0]]
        
        # Iterate through the predicted phonemes
        for phoneme in self.predicted_phonemes[1:]:
            # If the current phoneme is different from the last phoneme in the processed list
            # or it's a space (representing a pause), append it to the processed list
            if phoneme != processed_phonemes[-1] or phoneme == '-':
                processed_phonemes.append(phoneme)
                
        # Convert the list of processed phonemes to a string
        phonemes_string = ' '.join(processed_phonemes)
        
        # Create a response to be sent to the main application
        response = {
            "responseType": "multiplePhonemes",
            "body": {
                "classes": phonemes_string,
                "idStart": self.phoneme_id - len(self.predicted_phonemes),
                "idEnd": self.phoneme_id - 1
            }
        }

        # Clear the predicted phonemes list
        self.predicted_phonemes = []

        # Add the response to the queue
        self.predict_queue.put(json.dumps(response))