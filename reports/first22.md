# First 22 phonemes

Report for the data processing and machine learning done with the first 22 of 44 phonemes.

## <u>Contents</u>

- [EMG Data Processing](#emg-data-processing)
- [Data Processing Results and Analysis](#data-processing-results)
- [Machine Learning - LSTM RNN Model](#lstm-rnn-model)
- [Machine Learning Results and Analysis](#machine-learning-results)

## <a id="emg-data-processing" style="color: inherit; text-decoration: none;"><u>EMG Data Processing</u></a>

1. Import Libraries
    - Import necessary libraries for data processing and visualization (numpy, pandas, matplotlib).

2. Set Hyperparameters
    - Define the hyperparameters of the dataset, including phonemes, channels, window size, and step size.

3. Define `plot_all_channels` function
    - Create a function that takes a DataFrame and title as inputs and plots a figure with subplots for each channel.

4. Load and Preprocess Data
    - Load the CSV data into a Pandas DataFrame.
    - Filter out unnecessary columns and assign new column names.
    - Remove the first 4000 rows due to bad data.

5. Plot Raw EMG Channels
    - Plot raw EMG channels using the `plot_all_channels` function.

6. Outliers Removal and Data Normalization
    - Remove outliers by replacing them with the mean of surrounding values.
    - Normalize the data using Min-Max normalization and round the values to 0 or 1.
    - Create a new DataFrame `df_normalized`.

7. Define `plot_rolling_channel_4` and `segment_stats` Functions
    - Create functions for plotting the rolling maximum of EXG Channel 4 with different ranges and for calculating statistics about the segments.

8. Add Rolling Max Column
    - Add a new column for the rolling maximum of EXG Channel 4 with a specific window size.

9. Identify Segments
    - Identify the start and end indices of segments with 1s in the rolling maximum of EXG Channel 4.
    - Get segments that need to be zeroed and filter out small segments.

10. Clean Recordings and Group Segments
    - Define functions to clean the data, group the segments, and create a dictionary with separated phonemes.

11. Generate Training Examples and Labels
    - Create functions to generate training examples and labels, concatenate them into arrays, and save them as numpy files.

## <a id="data-processing-results" style="color: inherit; text-decoration: none;"><u>Data Processing Results and Analysis</u></a>
### **Hyperparameter Choices**
| Hyperparameter | Value     |
|----------------|-----------|
| PHONEMES       | /_, /p, /b, /t, /d, /k, /g, /f, /v, /s, /z, /m, /i, /ē, /e, /a, /u, /oo, /ū, /a(r), /ā, /ī, /oy |
| CHANNELS       | 3         |
| WINDOW_SIZE    | 10        |
| STEP_SIZE      | 10         |

### **Hyperparameter Methodology**
#### **Phonemes**
There are 22 phonemes (and a silence class), chosen (using GPT-4) for their EMG signal differences and muscle group activations.

#### **Channels**
There are 3 channels for 3 muscle groups: depressor labii inferioris (DLI), orbicularis oris superior (OOS), and orbicularis oris inferior (OOI). Attempt at recording with 7 channels failed (see `Project_Methods.png`), recorded with 4, but had to get rid of the zygomaticus major (ZYG) channel because of electorode connection problems. (Less channels than for 5 phoneme model)

#### **Window Size and Step Size**
To make the most of the available data, we experimented with creating datasets with various values of window size and step size, ranging from 10-50 and 1-50 respectively. We settled on 10 and 10 (meaning 40ms windows with no overlap), as this combination provides the following benefits:

1. **More training examples:** By breaking down the available data into smaller units, we can increase the number of training examples that are created from the same dataset.
2. **Faster model response time:** Using smaller window size and step size datasets allows the ML model to have a quicker response time when it is integrated into a real-time application.

### **Phoneme Recordings**
| Statistic                     | Value    |
|-------------------------------|-----------|
| Number of segments            | 231      |
| Average segment length (values)| 97.2     |
| Minimum segment length (values)| 50       |
| Maximum segment length (values)| 207      |
| Average segment length (ms)   | 388.81   |
| Minimum segment length (ms)   | 200      |
| Maximum segment length (ms)   | 828      |

### **Data Issues**

The dataset contains only 231 recordings in total, which is not ideal for training a robust machine learning model, and the distribution of phoneme training examples is not uniform, some phonemes have more representation than others. This limitation is a result of issues during the data collection process. It is important to note that the actual recordings included in the dataset are longer than necessary for an effective ML model. However, with the chosen window size and step size, we can create more training examples to train a better, more reactive and robust model.

### **Training Example Generation**
| Phoneme       | Examples   |
|---------------|------------|
| silence       | 100        |
| p             | 148        |
| b             | 64         |
| t             | 94         |
| d             | 80         |
| k             | 69         |
| g             | 95         |
| f             | 85         |
| v             | 90         |
| s             | 91         |
| z             | 114        |
| m             | 122        |
| i             | 91         |
| ē             | 103        |
| e             | 90         |
| a             | 102        |
| u             | 104        |
| oo            | 119        |
| ū             | 82         |
| a(r)          | 97         |
| ā             | 81         |
| ī             | 99         |
| oy            | 122        |

### **Final Training Data**
| Python Code              | Shape                 |
|--------------------------|-----------------------|
| X_train.shape            | (2242, 3, 10)         |
| y_train.shape            | (2242,)               |

### **Conclusion**

We have processed and analyzed our data of the selected 22 phonemes, gaining insight into the recording sessions and informing future decisions. Despite the limitations in data quality and quantity, the analysis and data processing approach enabled us to create a dataset suitable for a machine learning model.

## <a id="lstm-rnn-model" style="color: inherit; text-decoration: none;"><u>Machine Learning - LSTM RNN Model</u></a>

1. Import Libraries
    - Import necessary libraries for data processing, model building, and visualization (os, re, random, numpy, tensorflow, keras, scikit-learn, matplotlib, seaborn).

2. Set Hyperparameters
    - Define the hyperparameters of the dataset, including phonemes and classes.

3. Define `get_files_with_prefix` function
    - Create a function that takes a directory and prefix as inputs and returns a list of files with the specified prefix.

4. Load and Preprocess Data
    - Load the training data and parse the filenames.
    - Split the data into training and testing sets.
    - Prepare the data for input into the model.

5. Define the Model
    - Create a sequential model using various layers such as Conv1D, Dropout, LSTM, and Dense.
    - Compile the model with loss, optimizer, and metrics.

6. Define Early Stopping callback
    - Define an EarlyStopping callback function to prevent overfitting.

7. Train the Model
    - Fit the model on the training data and validate it on the testing data using the early stopping callback.

8. Evaluate the Model
    - Evaluate the final model on the testing set and display the test accuracy.

9. Plotting Training and Validation Loss and Accuracy
    - Create plots for training and validation loss, as well as training and validation accuracy.

10. Calculate and Display Confusion Matrix
    - Get the predicted classes for the entire test set.
    - Create and display the confusion matrix using a heatmap.

11. Test Model on Random Examples
    - Choose random examples from the test set and display their actual and predicted phonemes.

## <a id="machine-learning-results" style="color: inherit; text-decoration: none;"><u>Machine Learning Results and Analysis</u></a>
### **Train and Test Shapes**
| Python Code              | Shape             |
|--------------------------|-------------------|
| X_train.shape            | (1793, 3, 10)     |
| X_test.shape             | (449, 3, 10)      |

### **Data Preparation and Considerations**

We prepared the dataset using an 80/20 split, which resulted in an appropriate total count and phoneme distribution for both the training and test sets. It is important to take into consideration the fact that using a sliding window teachnique has generated training examples that may be adjacent and/or from the same phoneme recording that end up in both the training and test set, which could lead to data leakage. The choice of parameters, 10 and 10 (40 ms), was intentional, in order to have zero overalap while also creating enough training examples that could be used to make a real-time transcription model.

### **Model Summary**
Model: "sequential_6"  
Total params: 837,783
| Layer (type)                | Output Shape       | Param Count   |
|-----------------------------|--------------------|---------------|
| conv1d_12 (Conv1D)          | (None, 3, 128)     | 3968          |
| batch_normalization_8       | (None, 3, 128)     | 512           |
| dropout_8 (Dropout)         | (None, 3, 128)     | 0             |
| conv1d_13 (Conv1D)          | (None, 2, 256)     | 98560         |
| batch_normalization_9       | (None, 2, 256)     | 1024          |
| dropout_9 (Dropout)         | (None, 2, 256)     | 0             |
| lstm_12 (LSTM)              | (None, 2, 256)     | 525312        |
| batch_normalization_10      | (None, 2, 256)     | 1024          |
| dropout_10 (Dropout)        | (None, 2, 256)     | 0             |
| lstm_13 (LSTM)              | (None, 256)        | 525312        |
| batch_normalization_11      | (None, 256)        | 1024          |
| dropout_11 (Dropout)        | (None, 256)        | 0             |
| dense_6 (Dense)             | (None, 256)        | 65792         |
| dense_7 (Dense)             | (None, 22)         | 5657          |  

### **Evaluation Results on X_test**
| Evaluation Statistic        | Value                           |
|-----------------------------|---------------------------------|
| Test Example Count          | 449                             |
| Test Loss                   | 0.12511923909187317             |
| Test Accuracy               | 0.9910913109779358 (445/449)    |

### **Evaluation Conclusions**

These results demonstrate that the model effectively recognizes phonemes with high accuracy while exhibiting low loss levels. However, the model might be overfitting considering the limited data and quick achievement of 95%+ accuracy. For more phonemes, a larger model size could help scale the number of classes, or a change in hyperparameters, algorithms, or layers.