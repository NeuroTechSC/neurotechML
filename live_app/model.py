import threading
import time
import logging
import pandas as pd
import numpy as np
from queue import Queue
from collections import deque
from tensorflow.keras.models import load_model
import helper

## GLOBALS
data_queue = Queue() # Queue to store data indices
processed_data_queue = Queue() # Queue to store processed data
predictions_queue = helper.buffer_queue # Queue to store predictions

PHONEMES = ['_', 'B', 'D', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'Y', 'Z', 'CH', 'SH', 'NG', 'DH', 'TH', 'ZH', 'WH', 'AA', 'AI(R)', 'I(R)', 'A(R)', 'ER', 'EY', 'IY', 'AY', 'OW', 'UW', 'AE', 'EH', 'IH', 'AO', 'AH', 'UH', 'OO', 'AW', 'OY']
WINDOW_SIZE = 5  # Number of data points to be used for prediction
    
EMG_DATA = pd.DataFrame() # DataFrame to store EMG data
CURRENT_PHONEMES = [] # List to store current phonemes

class DataThread(threading.Thread):
    def __init__(self, threadID, csv_file, model_path):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.daemon = True  # The thread will exit when the main program exits
        self.csv_file = csv_file  # Mock CSV file path
        self.model_path = model_path  # LSTM/RNN model path

        # Logging
        
        # Initialize the logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(ch)
        
        self.logger.info(f"Thread {self.threadID} initialized.")

    def run(self):
        # Create threads
        stream_thread = TestStreamThread(1.1, self.csv_file, self.logger)
        process_thread = ProcessingThread(1.2, self.logger)
        predict_thread = PredictionThread(1.3, self.model_path, self.logger)

        # Start Threads
        if helper.ready_state.is_set():
            stream_thread.start()
            process_thread.start()
            predict_thread.start()

# class OpenBCIThread(threading.Thread):
#     def __init__(self, threadID, csv_file, logger):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.daemon = True  # The thread will exit when the main program exits
#         self.csv_file = csv_file  # Mock CSV file path

#         WINDOW_SIZE = 5  # Number of data points to be used for prediction
#         self.running_state = False # State of the button (True = running, False = stopped)
        
#         self.logger = logger
#         self.logger.info(f"Thread {self.threadID} initialized.")

class TestStreamThread(threading.Thread):
    def __init__(self, threadID, csv_file, logger):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.daemon = True  # The thread will exit when the main program exits
        self.csv_file = csv_file  # Mock CSV file path

        self.running_state = False # State of the button (True = running, False = stopped)
        self.memory_limit = 1e7 # Memory limit for the DataFrame
        self.chunks_ingested = 0 # Number of chunks ingested
        self.current_chunk = 0 # Current chunk being processed
        
        self.logger = logger
        self.logger.info(f"Thread {self.threadID} initialized.")

    def run(self):
        self.logger.info(f"Thread {self.threadID} started.")

        global EMG_DATA

        column_names = [' EXG Channel 0', ' EXG Channel 1', ' EXG Channel 2', ' EXG Channel 3']
        new_column_names = ['DLI', 'OOS', 'OOI', 'PLA']

        # Read the entire file once
        data = pd.read_csv(self.csv_file, skiprows=4, usecols=column_names)
        
        # Rename the columns
        data.columns = new_column_names

        # Skip the unwanted rows at the beginning of the data
        data = data[10:]

        while True:
            self.logger.info(f"Ingesting data chunk {self.chunks_ingested}.")

            if len(data) < WINDOW_SIZE:
                self.logger.info('Not enough data. Stopping thread.')
                return

            # Get a chunk of data
            data_chunk = data[:WINDOW_SIZE]

            # Remove the processed rows from the original data
            data = data[WINDOW_SIZE:]
            
            # Append the data to the DataFrame
            if EMG_DATA.empty:
                EMG_DATA = data_chunk
            else:
                EMG_DATA = pd.concat([EMG_DATA, data_chunk])

            # If DataFrame memory usage exceeds limit, drop the oldest data
            if EMG_DATA.memory_usage(index=True, deep=True).sum() > self.memory_limit:
                EMG_DATA = EMG_DATA[WINDOW_SIZE:]
                self.current_chunk -= 1

            # If the button is in the 'running' state
            if helper.button_state.is_set():
                self.logger.info("Button is in 'running' state.")

                # Update the button state
                self.running_state = True

                # Put the chunk indices in the queue
                helper.queuePut(data_queue, self.current_chunk)

            # If the button just in the 'running' state
            elif self.running_state:
                self.logger.info("Button just set to 'stopped' state.")

                # Create a thread to process the final phonemes
                final_processing_thread = FinalProcessing(4, self.logger)

                # Start the thread
                final_processing_thread.start()

                # Update the button state
                self.running_state = False

            # If the button is and has been in the 'stopped' state
            else:
                self.logger.info("Button is in 'stopped' state.")

            # Increment the number of chunks ingested
            self.chunks_ingested += 1
            self.current_chunk += 1

            # 4ms pause to simulate real-time data streaming
            time.sleep(0.004)  

class ProcessingThread(threading.Thread):
    def __init__(self, threadID, logger):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.daemon = True  # The thread will exit when the main program exits
        
        self.logger = logger
        self.logger.info(f"Thread {self.threadID} initialized.")

    def run(self):
        self.logger.info(f"Thread {self.threadID} started.")

        while True:
            # Get the chunk index from the queue
            index = helper.queueGet(data_queue)

            self.logger.info(f"Processing data chunk {index}.")

            # Normalize the data using Min-Max normalization
            df_normalized = (EMG_DATA - EMG_DATA.min()) / (EMG_DATA.max() - EMG_DATA.min())

            # Get the data chunk
            data_chunk = df_normalized.iloc[index:index + WINDOW_SIZE]

            # Put the processed data in the queue
            helper.queuePut(processed_data_queue, {index : data_chunk})

class PredictionThread(threading.Thread):
    def __init__(self, threadID, model_path, logger):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.daemon = True  # The thread will exit when the main program exits

        self.model = load_model(model_path) # Load LSTM/RNN model

        helper.ready_state.set() # Set the ready state
        
        self.logger = logger
        self.logger.info(f"Thread {self.threadID} initialized.")
    
    def run(self):
        self.logger.info(f"Thread {self.threadID} started.")

        global CURRENT_PHONEMES

        while True:
            # Get the processed data from the queue
            data = helper.queueGet(processed_data_queue)

            # Predict phoneme
            phoneme = self.predict_phoneme(list(data.values())[0])

            self.logger.info(f"Predicted phoneme {list(data.keys())[0]}: {phoneme}.")

            # Create a response to be sent to the main application
            response = {
            "responseType": "singlePhoneme",
            "body": {
                "class": phoneme,
                "id": list(data.keys())[0],
                "row": 4,
                "col": WINDOW_SIZE,
                "data": list(data.values())[0]
                }
            }

            # Add the response to the queue
            helper.queuePut(predictions_queue, response)

            # Add the phoneme to the list of current phonemes
            CURRENT_PHONEMES.append(phoneme)

    def predict_phoneme(self, input_data):
        # Predict class softmax probabilities
        prediction = self.model.predict(np.array(input_data).reshape(1, 4, 5), verbose=0)

        # Return the phoneme with the highest probability
        return PHONEMES[np.argmax(prediction, axis=1)[0]]

class FinalProcessing(threading.Thread):
    def __init__(self, threadID, logger):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.daemon = True

        self.logger = logger

    def run(self):
        self.logger.info("Processing phonemes.")

        global CURRENT_PHONEMES

        self.logger.info(f"Phoneme list: {CURRENT_PHONEMES}")

        # Initialize the processed list with the first phoneme
        processed_phonemes = [CURRENT_PHONEMES[0]]
        
        # Iterate through the predicted phonemes
        for phoneme in CURRENT_PHONEMES[1:]:
            # If the current phoneme is different from the last phoneme in the processed list
            # or it's a space (representing a pause), append it to the processed list
            if phoneme != processed_phonemes[-1] or phoneme == '-':
                processed_phonemes.append(phoneme)
                
        # Convert the list of processed phonemes to a string
        phoneme_string = ' '.join(processed_phonemes)

        self.logger.info(f"phoneme_string: {phoneme_string}")
        
        # Create a response to be sent to the main application
        response = {
            "responseType": "multiplePhonemes",
            "body": {
                "classes": phoneme_string,
                "idStart": 0,
                "idEnd": len(phoneme_string)
            }
        }

        # Add the response to the queue
        helper.queuePut(predictions_queue, response)

        # Write the response to a file
        with open('predictions.txt', 'a') as f:
            f.write(f"{CURRENT_PHONEMES}\n")
            f.write(f"{phoneme_string}\n\n")

        # Clear the predicted phonemes list
        CURRENT_PHONEMES = []
