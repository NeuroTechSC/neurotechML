import threading
import time
from model import ThreadClass
from keras.models import load_model

print("Started example_main.py")

# Load the Keras model
model_path = "LSTM_all44_seed489_5_5_568k"

print("Loaded model")

# Create a button event
button_event = threading.Event()

# Create an instance of ThreadClass
thread = ThreadClass(1, "data_all44.csv", model_path, button_event)

print("Created thread")

# Start the thread
thread.start()

print("Started thread")

# To change the button state, use the `set` and `clear` methods of the button_event
# To start the data collection and processing, call `set`
button_event.set()

print("Button started")

# Wait for some time (this will depend on your specific use case)
time.sleep(5)

# To stop the data collection and processing, call `clear`
button_event.clear()

print("Button stopped")

time.sleep(0.5)

print("example_main.py ended")