import threading
import time
from model import DataThread
from helper import button_state

if __name__ == "__main__":

    print("in main.py")

    # Create threads
    data_thread = DataThread(1, "data/data_all44.csv", "LSTM_all44_seed489_5_5_568k")
    # gui_thread = gui_c(2)

    print("Created threads")

    # Start Threads
    data_thread.start()
    # gui_thread.run()

    print("Started data_thread")

    time.sleep(5)

    # To change the button state, use the `set` and `clear` methods of the button_event
    # To start the data collection and processing, call `set`
    button_state.set()

    print("button_state set")

    # Wait for some time (this will depend on your specific use case)
    time.sleep(10)

    # To stop the data collection and processing, call `clear`
    button_state.clear()

    print("button_state cleared")

    time.sleep(5)

    print("main.py ended")