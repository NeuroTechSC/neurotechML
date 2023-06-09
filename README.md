# NeuroTechSC - Machine Learning

This is the Git repository for the Machine Learning team of the NeuroTechSC organization. The main goal of our project is to detect (subvocal) phonemes from surface Electromyography (sEMG) data using a Deep Neural Network model implemented in Python. This repository provides the Python notebooks for the sEMG data preprocessing and various model trainings.

## <u>Contents</u>

- [General Research Abstract for Project](#research-abstract)
- [Machine Learning Plan Overview](#ml-plan-overview)
- [Project Timeline](#project-timeline)
- [File Descriptions](#file-descriptions)
- [EMG Data Processing](#emg-data-processing)
- [Data Processing Results and Analysis](#data-processing-results)
- [Machine Learning - LSTM RNN Model](#lstm-rnn-model)
- [Machine Learning Results and Analysis](#machine-learning-results)

## <a id="research-abstract" style="color: inherit; text-decoration: none;"><u>General Research Abstract for Project</u></a>

Subvocalization refers to the internal speech that occurs while reading or thinking without producing any audible sound. It is accompanied by the activation of the muscles involved in speech production, generating electrical signals known as electromyograms (EMGs). These signals can potentially be used for silent communication and assistive technologies, improving the lives of people with speech impairments and enabling new forms of human-computer interaction. However, the accurate recognition of subvocal EMGs remains a challenge due to the variability in signal patterns and electrode placement. Our proposed solution is to develop an advanced subvocal EMG recognition system using machine learning techniques. By analyzing the EMG signals, our system will be able to identify the intended speech content and convert it into text. This technology will be applicable in various fields, including silent communication for military or emergency personnel, assistive devices for people with speech impairments, and hands-free control of computers and other electronic devices.

## <a id="ml-plan-overview" style="color: inherit; text-decoration: none;"><u>Machine Learning Plan Overview (Original)</u></a>

This project aims to improve the performance of subvocal phoneme detection using machine learning techniques. Subvocal phonemes are the speech components generated when a person talks to themselves without producing any sound. Detecting these phonemes has various applications, including silent communication devices and assistive technologies for individuals with speech impairments.

The TMC-ViT model used in this repository is a novel deep learning architecture that leverages the benefits of vision transformers and temporal multi-channel features to achieve improved performance on sEMG data. This model outperforms conventional methods such as CNNs and LSTMs in subvocal phoneme detection tasks. 

We will first train our model on data where the phonemes are audible and voiced, then transition to whispered, and eventually we hope to be able to detect subvocal speech (like in reading or intense thought).

### **Machine Learning Update I**

Now using a LSTM/RNN model for phoneme recognition instead of the TMC-ViT, check `gtp_convos/gpt_convo_2.md` for more information. Replaced `TMC-ViT.ipynb` with `LSTM_RNN_bviou.ipynb`.

### **Machine Learning Update II**

Expanded from initial five phonemes `bviou` to a diverse set of 22. Improved data processing and precision, increased model size and robustness, and added guards against overfitting.

### **Machine Learning Update III**

Recorded complementary set of phonemes (last 22), changed muscle groups, greatly improved data quality! Accounted for more robust/diverse silence/rest class in processing, upped model parameter count, got great performance. Added model performance analysis section to data processing notebook (see `LSTM_RNN_last_22.ipynb`).

## <a id="project-timeline" style="color: inherit; text-decoration: none;"><u>Project Timeline</u></a>

1. Came up with a design plan and chose model architecture (TMC-ViT)
2. Collected data for 5 phonemes
3. Finished preprocessing data
4. Analyzed data and presented findings at the California Neurotech Conference on April 29th, 2023 (see Research Abstract section)
5. Assessed model viability and are considering a pivot from TMC-ViT to an LSTM/RNN network
6. Created training examples with various hyperparameters (see `data/`)
7. Pivoted to LSTM/RNN architecture, achieved near 100% accuracy on test data with ~155,000 parameters (see `archive/LSTM_RNN_bviou.ipynb`)
8. Expanded from initial five phonemes `bviou` to a diverse set of 22, attempted recording with 7 channels/muscle groups (attempt failed, only 3 usable channels)
9. Processed data and modified model hyperparameters to scale to more classes, achieved ~99% accuracy on test data with ~836,000 parameters (see `LSTM_RNN_first_22.ipynb`)
10. Collected data for remaining 22 phonemes, used 4 new muscle groups/channels, quality recordings got processed into super clean data
11. Trained model which achieved ~98.3% accuracy on test data with 1,373,591 parameters (see `LSTM_RNN_last_22.ipynb`)
12. Added section to data processing notebook which tests model on entire original recording (see `EMG_Data_Processing_last_22.ipynb`)

### **Next Steps**
1. Re-record first 22 phonemes with new muscle groups
2. Assess the viability of a second model to correct phoneme/letter-level errors (phoneme list to word string model?)
3. Build a real-time transcription app
4. Upgrade model to whispered/subvocal phoneme recognition.

## <a id="file-descriptions" style="color: inherit; text-decoration: none;"><u>File Descriptions</u></a>

- `EMG_Data_Processing_first_22.ipynb` - preprocessing script that cleans the data for the first 22 phonemes and formats it into training examples
- `EMG_Data_Processing_last_22.ipynb` - preprocessing script that cleans the data for the last 22 phonemes and formats it into training examples
- `LSTM_RNN_first_22.ipynb` - a model that reached ~99% test accuracy on 22 diverse phonemes, has training code + visualization
- `LSTM_RNN_last_22.ipynb` - a model that reached ~98.3% test accuracy on the other set of 22 diverse phonemes and a silence/rest class, has training code + visualization
- `Project_Methods.png` - image showing history of recording sessions 
- `git_push_script.bat` - batch script to quickly push all changes to the repository with a commit message
- `archive/` - folder which contains old model training and data processing notebooks
- `data/` - folder which contains the raw .csv files from the recordings, as well as formatted training example/label .npy files
- `gtp_convos/` - discussions with GPT-4 about the project
- `models/` - folder which contains saved models, not just weights
- `pictures/` - folder which contains pictures for the README.md
- `reports/` - folder which contains reports on Python notebooks, markdown reports of data processing and machine learning results and analysis

## <a id="file-descriptions" style="color: inherit; text-decoration: none;"><u>Results</u></a>

TBA - Should have final (first 22) phonemes recorded by 6/10/2023, and model created within a few days of that, results will be put here.