import Tkinter as tk

class SidebarFrame():

    def __init__(self, root):

        # NeuroTechSC
        tk.Label(root, text="EMG Phoneme Prediction").grid(row=0, column=0, sticky="nsew")
        tk.Label(root, text="NeuroTechSC").grid(row=1, column=0, sticky="nsew")

        # Buttons
        exitButton = tk.Button(root, text="Exit").grid(row=9, column=0)
        downloadButton = tk.Button(root, text="Download Data").grid(row=5, column=0, sticky="nsew")
        resetModelButton = tk.Button(root, text="resetModel").grid(row=6, column=0, sticky="nsew")

