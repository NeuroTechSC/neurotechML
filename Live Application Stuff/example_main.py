import threading
import time
from model import ThreadClass, ButtonEvent
from keras.models import load_model

# Load the Keras model
model_path = "model_path"

# Create a button event
button_event = ButtonEvent()

# Create an instance of ThreadClass
thread = ThreadClass(1, "openbci_data.csv", model_path, button_event)

# To change the button state, use the `set` and `clear` methods of the button_event
# To start the data collection and processing, call `set`
button_event.set()

# Wait for some time (this will depend on your specific use case)
time.sleep(10)

# To stop the data collection and processing, call `clear`
button_event.clear()