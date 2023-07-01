import Tkinter as tk
from helper import *
import threading
from sidebarFrame import SidebarFrame
from predictionFrame import predictionFrame
from bottomFrame import BottomFrame

def click(stringVar):
    stringVar.set("Hello, Another!")


class gui_c(threading.Thread):

    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.i = 0

        self.root = tk.Tk()
        self.root.title("EMG Phoneme Prediction")
        self.root.geometry("1024x720")
        self.root.grid_propagate(False)

        # Sidebar Frame
        sidebar = SidebarFrame(self.root)

        predictionFrames = predictionFrame(self.root)

        bottomFrame = BottomFrame(self.root)

    def update(self):


        # Keep this last
        self.root.after(1, self.update)
        return
    
    def run(self):
        self.root.after(1, self.update)
        self.root.mainloop()