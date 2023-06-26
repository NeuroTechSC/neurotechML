import threading
import queue
HOST = 'localhost'
PORT = 8080

## GLOBALS
buffer_queue = queue.Queue(0)

def queuePut(buffer, item):
    buffer.put(item)

def queueGet(buffer):
    return buffer.get()

# button state
button_state = threading.Event() # 'stopped' / 'running' (unset, set)
reset_model = threading.Event() # 'set means to reset model
ready_state = threading.Event() # HARD CODE A 10s timer for the user instead of this Event variable (set means to reset)