import Tkinter as tk

class BottomFrame():

    def __init__(self,root):

        self.statusText = tk.StringVar(root, value="Ready")
        self.statusLabel = tk.Label(root, textvariable=self.statusText, bg="green", justify="center", fg="white")

        self.statusLabel.grid(row=5, column=1, columnspan=8, sticky="nsew")

        # Start/Stop button
        self.startButton = tk.Button(root, text="START")
        self.startButton.grid(row=7, column=2)

        ## Correct / Incorrect Buttons
        self.correctBtn = tk.Button(root, text="Correct", state="disabled")
        self.correctBtn.grid(row=8, column=1)
        self.incorrectBtn = tk.Button(root, text="Incorrect", state="disabled")
        self.incorrectBtn.grid(row=8, column=2)

        # Metrics (Right Side), Col > 5, row > 5
        self.accuracy = 0.0
        self.accuracyText = tk.StringVar(root, value="Accuracy: 0%")
        tk.Label(root,textvariable=self.accuracyText).grid(row=7, column=6)
