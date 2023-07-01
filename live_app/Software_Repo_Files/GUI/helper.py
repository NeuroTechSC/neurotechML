import threading
HOST = 'localhost'
PORT = 8080

## GLOBALS
buffer = []
current_item = None
button_state = 'stopped'

lock = threading.Lock()

def buffer_push(buffer, item):
    with lock:
        buffer.append(item)
    return

def buffer_pop(buffer):
    json_ret = None
    with lock:

        if len(buffer) != 0:
            json_ret = buffer[0]
            del buffer[0]
    return json_ret

lock2 = threading.Lock()
def toggleButton():
    with lock2:
        if button_state == 'running':
            button_state = 'stopped'
        else:
            button_state = 'running'

def readButton():
    with lock2:
        return button_state