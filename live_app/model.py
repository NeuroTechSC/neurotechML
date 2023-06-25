import threading
import time
import json
import logging
import pandas as pd
import numpy as np
from queue import Queue
from collections import deque
from tensorflow.keras.models import load_model

PHONEMES = ['_', 'B', 'D', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'Y', 'Z', 'CH', 'SH', 'NG', 'DH', 'TH', 'ZH', 'WH', 'AA', 'AI(R)', 'I(R)', 'A(R)', 'ER', 'EY', 'IY', 'AY', 'OW', 'UW', 'AE', 'EH', 'IH', 'AO', 'AH', 'UH', 'OO', 'AW', 'OY']

class ThreadClass(threading.Thread):
    def __init__(self, threadID, csv_file, model_path, button_event, window_size=5, memory_limit=1e6):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.daemon = True  # The thread will exit when the main program exits
        self.csv_file = csv_file  # Mock CSV file path
        self.model = load_model(model_path)  # LSTM/RNN model
        self.memory_limit = memory_limit  # Memory limit for the DataFrame
        self.button_event = button_event # Event to trigger phoneme prediction
        self.emg_data = pd.DataFrame()  # DataFrame to store EMG data
        self.predicted_phonemes = []  # List to store predicted phonemes
        self.lock = threading.Lock()  # Lock for thread-safe operations
        self.predict_queue = Queue()  # Queue to store predictions
        self.window_size = window_size  # Number of data points to be used for prediction
        self.phoneme_id = 0 # ID for each phoneme prediction
        self.button_state = False # State of the button (True = running, False = stopped)

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

        column_names = [' EXG Channel 0', ' EXG Channel 1', ' EXG Channel 2', ' EXG Channel 3']
        new_column_names = ['DLI', 'OOS', 'OOI', 'PLA']

        # Read the entire file once
        data = pd.read_csv(self.csv_file, skiprows=4, usecols=column_names)
        
        # Rename the columns
        data.columns = new_column_names

        # Skip the unwanted rows at the beginning of the data
        data = data[10:]

        while True:
            if len(data) < self.window_size:
                logging.info('Not enough data. Stopping thread.')
                return

            # Get a chunk of data
            data_chunk = data[:self.window_size]

            # Remove the processed rows from the original data
            data = data[self.window_size:]
            
            # Append the data to the DataFrame
            self.emg_data = pd.concat([self.emg_data, data_chunk])
            
            # Record the current timestamp in the queue
            timestamp_queue.append(time.time())

            # If the button is in the 'running' state
            if self.button_event.is_set():
                self.logger.info("Button is in 'running' state.")

                # Update the button state
                self.button_state = True

                # Predict phoneme
                self.predict_phoneme()
                
                # If DataFrame memory usage exceeds limit, drop the oldest data
                if self.emg_data.memory_usage(index=True, deep=True).sum() > self.memory_limit:
                    self.emg_data = self.emg_data.iloc[self.window_size:]

            # 
            elif self.button_state:
                self.logger.info("Button was set to 'stopped' state.")

                # Process phonemes
                self.process_phonemes()

                # Update the button state
                self.button_state = False

            else:
                self.logger.info("Button is in 'stopped' state.")
            
            self.logger.info("Waiting for next phoneme.")
            
            # Pause before next iteration
            time.sleep(0.004)  # 4ms pause to simulate real-time data streaming

    def predict_phoneme(self):
        self.logger.info("Predicting phoneme...")

        # Get the most recent data for prediction
        recent_data = self.emg_data.tail(self.window_size)
        
        # Preprocess the data
        input_data = self.preprocess(recent_data)

        # Predict phoneme
        # print(np.array(input_data).reshape(4, 5).shape)
    
        prediction = self.model.predict(np.array(input_data).reshape(1, 4, 5), verbose=0)

        # print(np.argmax(prediction, axis=1))

        phoneme = PHONEMES[np.argmax(prediction, axis=1)[0]]

        self.logger.info(f"Predicted phoneme: {phoneme} | {self.phoneme_id}")
        
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

        self.logger.info(f"predicted_phonemes: {self.predicted_phonemes}")

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

        self.logger.info(f"phonemes_string: {phonemes_string}")
        
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