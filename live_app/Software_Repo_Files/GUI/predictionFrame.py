import Tkinter as tk

class predictionFrame():

    def __init__(self, root):

        self.phonemePredictionText = tk.StringVar(root, value="Click START to begin...")
        self.wordPredictionText = tk.StringVar(root, value="Click START to begin...")

        self.phonemeLabel = tk.Label(root, textvariable=self.phonemePredictionText, bg='gray', width=50, height=20)

        self.wordLabel = tk.Label(root, textvariable=self.wordPredictionText, width=50, height=20)

        self.phonemeLabel.grid(row=0, column=1, rowspan=5, columnspan=4, sticky="nsew")
        self.phonemeLabel.grid_propagate(False)

        self.wordLabel.grid(row=0, column=5, rowspan=5, columnspan=4, sticky="nsew")
        self.wordLabel.grid_propagate(False)