# NeuroTechSC - Machine Learning

This is the Git repository for the Machine Learning team of the NeuroTechSC organization. The main goal of our project is to detect subvocal phonemes from surface Electromyography (sEMG) data using a Deep Neural Network model implemented in Python. This repository provides the Python notebooks for the sEMG data preprocessing and various model trainings.

## General Research Abstract for Project

Subvocalization refers to the internal speech that occurs while reading or thinking without producing any audible sound. It is accompanied by the activation of the muscles involved in speech production, generating electrical signals known as electromyograms (EMGs). These signals can potentially be used for silent communication and assistive technologies, improving the lives of people with speech impairments and enabling new forms of human-computer interaction. However, the accurate recognition of subvocal EMGs remains a challenge due to the variability in signal patterns and electrode placement. Our proposed solution is to develop an advanced subvocal EMG recognition system using machine learning techniques. By analyzing the EMG signals, our system will be able to identify the intended speech content and convert it into text. This technology will be applicable in various fields, including silent communication for military or emergency personnel, assistive devices for people with speech impairments, and hands-free control of computers and other electronic devices.

## Machine Learning Plan Overview (Original)

This project aims to improve the performance of subvocal phoneme detection using machine learning techniques. Subvocal phonemes are the speech components generated when a person talks to themselves without producing any sound. Detecting these phonemes has various applications, including silent communication devices and assistive technologies for individuals with speech impairments.

The TMC-ViT model used in this repository is a novel deep learning architecture that leverages the benefits of vision transformers and temporal multi-channel features to achieve improved performance on sEMG data. This model outperforms conventional methods such as CNNs and LSTMs in subvocal phoneme detection tasks. 

## Machine Learning Plan Amendment

Now using a LSTM/RNN model for phoneme recognition instead of the TMC-ViT, check `gtp_convos/gpt_convo_2.md` for more information. Replaced `TMC-ViT.ipynb` with `LSTM_RNN.ipynb`.

## Project Timeline

1. Came up with a design plan and chose model architecture (TMC-ViT)
2. Collected data for 5 phonemes
3. Finished preprocessing data
4. Analyzed data and presented findings at the California Neurotech Conference on April 29th, 2023 (see Research Abstract section)
5. Assessed model viability and are considering a pivot from TMC-ViT to an LSTM/RNN network
6. Created training examples with various hyperparameters (see `data/`)
7. Pivoted to LSTM/RNN architecture, achieved near 100% accuracy on test data with ~160,000 parameters (see `LSTM_RNN.ipynb`)

### Next Steps

1. Collect data on more phonemes, upgrade Cyton OpenBCI board to support 7 muscle groups instead of 4
3. Assess the viability of a second model to correct phoneme/letter-level errors (phoneme list to word string model)
4. Build a real-time transcription app

## File Descriptions

- `LSTM_RNN.ipynb` - a model that reached ~100% test accuracy on our 5 phonemes, has training code + visualization
- `EMG_Data_Processing.ipynb` - preprocessing script that cleans the data and formats it into training examples
- `gtp_convos/gpt_convo.md` - a discussion on processing .adc files and modifying TMC-ViT code
- `gtp_convos/gpt_convo_2.md` - a discussion on using transformer and RNN architectures for subvocal phoneme prediction
- `data/` - folder which contains the raw .csv files from the recordings, as well as formatted example/label .npy files

## EMG_Data_Processing.ipynb

1. Import the necessary libraries, such as numpy, pandas, matplotlib.pyplot, random, and os.

2. Set hyperparameters for phonemes, channels, size, and step.

3. Define a function plot_all_channels that takes a DataFrame and a title as inputs. It creates a figure with subplots for each channel.

4. Load and preprocess the CSV data into a Pandas DataFrame, skipping the first few lines of metadata, and rename the columns by removing leading spaces.

5. Plot the raw "EXG" channels using the plot_all_channels function.

6. Normalize the data using Min-Max normalization, round "EXG Channel 4" values to 0 or 1, and create a new DataFrame df_normalized.

7. Add a new column to the DataFrame for the rolling maximum of "EXG Channel 4" (initial window size of 90).

8. Define plot_rolling_channel_4 and segment_stats functions for plotting the rolling maximum of "EXG Channel 4" with different ranges and calculating statistics about the segments.

9. Identify the start and end indices of segments with 1s in the rolling maximum of "EXG Channel 4".

10. Plot and analyze different segments of the data using the plot_rolling_channel_4 and segment_stats functions.

11. Generate training examples for each phoneme, such as train_silence, train_b, train_v, train_i, train_u, and train_o, using the generate_training_examples function.

12. Concatenate the training arrays, create the X_train array and print its shape.

13. Generate training labels for each phoneme, such as y_train_silence, y_train_b, y_train_v, y_train_i, y_train_u, and y_train_o, using the np.full function.

14. Concatenate the training labels and create the y_train array. Print its shape.

15. Save the X_train and y_train arrays as NumPy files with an appropriate filename format.

The script processes the EMG data, normalizes it, segments it, generates training examples, and saves the resulting data for further use.

## Data Processing Results

```
| Statistic                | Values    | Time (ms)   |
|--------------------------|-----------|-------------|
| Number of segments       | 53        |             |
| Average segment length   | 292.53    | 1175.75     |
| Minimum segment length   | 102       | 407.0       |
| Maximum segment length   | 375       | 1495.0      |

| Training Data Info       | Length    |
|--------------------------|-----------|
| train_silence            | 649       |
| train_b                  | 367       |
| train_v                  | 718       |
| train_i                  | 669       |
| train_u                  | 638       |
| train_o                  | 635       |

| Data Shape               | Shape             |
|--------------------------|-------------------|
| X_train.shape            | (3676, 4, 10)     |
| y_train.shape            | (3676,)           |
```

## LSTM_RNN.ipynb

1. Import the required libraries and modules for data manipulation, machine learning and plotting (Numpy, TensorFlow, Keras, Scikit-learn, Matplotlib, Seaborn).

2. Define the phonemes to be used for classification, and read the input data files (X_filename and y_filename) from the specified directory.

3. Define and implement a function to parse the filename for retrieving the number of examples, channels, and size of input data.

4. Load the training data (X_train_og, y_train_og) and split it into training (X_train, y_train) and test (X_test, y_test) datasets using Scikit-learn's train_test_split function.

5. Prepare the data by re-shaping the input tensors to have size (N, CHANNELS, SIZE).

6. Count the occurrences of each class in the original, training, and test sets; print the phoneme counts.

7. Define a deep learning model with multiple layers (Conv1D, LSTM, Dense) using TensorFlow and Keras.

8. Compile the model using the 'sparse_categorical_crossentropy' loss, 'adam' optimizer, and 'accuracy' metric.

9. Define an Early Stopping callback to monitor the 'val_accuracy' with a patience value of 100.

10. Train the model using the fit function with the training and validation data, batch size of 128, and up to 300 epochs (initially).

11. Evaluate the model performance on the test set and print the test loss and accuracy scores.

12. Plot the training and validation loss, and training and validation accuracy, over time (epochs).

13. Display the confusion matrix using a heatmap, which compares the actual and predicted phonemes for the test dataset.

14. Choose 10 random examples from the test set and predict the phoneme using the trained model. Compare the actual and predicted phonemes for these examples.

## Machine Learning Results

