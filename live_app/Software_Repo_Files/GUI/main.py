from gui import *
from helper import *


if __name__ == "__main__":

    gui_thread = gui_c(2)

    # Start GUI
    gui_thread.run()