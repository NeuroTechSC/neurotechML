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

### **Machine Learning Update IV**

Had session to collect diverse set of recordings with new muscle groups: Depressor labii inferioris (DLI), Orbicularis oris superior (OOS), Orbicularis oris inferior (OOI), and the Platysma (PLA)

- all 44 phonemes, each said 10 times (`data/chris_final/data_all44.csv`)
- 10 test words broken into syllables (`data/chris_final/test_sentences/vocal_test_words.csv`)
- 10 test sentences (`data/chris_final/test_sentences/vocal_test_sentences.csv`)
- ABCs vocal/whispered (`data/chris_final/ABCs/vocal_and_whispered_ABCs.csv`)
- ABCs silent (`data/chris_final/ABCs/silent_ABCs.csv`)
- Research Abstract vocal/whispered (`data/chris_final/research_abstract/vocal_and_whispered_research_abstract.csv`)
- Research Abstract silent (`data/chris_final/research_abstract/silent_research_abstract.csv`)

## <a id="project-timeline" style="color: inherit; text-decoration: none;"><u>Project Timeline</u></a>

1. Came up with a design plan and chose model architecture (TMC-ViT) - `early 2023`
2. Collected data for 5 phonemes - `2023/04/19`
3. Finished preprocessing data - `2023/04/24`
4. Analyzed data and presented findings at the California Neurotech Conference on April 29th, 2023 (see Research Abstract section)
5. Assessed model viability and are considering a pivot from TMC-ViT to an LSTM/RNN network - `2023/05/14`
6. Created training examples with various hyperparameters (see `data/`)
7. Pivoted to LSTM/RNN architecture, achieved near 100% accuracy on test data with ~155,000 parameters (see `archive/LSTM_RNN_bviou.ipynb`)
8. Expanded from initial five phonemes `bviou` to a diverse set of 22, attempted recording with 7 channels/muscle groups (attempt failed, only 3 usable channels) - `2023/05/28`
9. Processed data and modified model hyperparameters to scale to more classes, achieved ~99% accuracy on test data with ~836,000 parameters (see `LSTM_RNN_first_22.ipynb`)
10. Collected data for remaining 22 phonemes, used 4 new muscle groups/channels, quality recordings got processed into super clean data - `2023/06/08`
11. Trained model which achieved ~98.3% accuracy on test data (window=10) with 1,373,591 parameters (see `LSTM_RNN_last_22.ipynb`)
12. Trained model which achieved ~99.0% accuracy on test data (window=5) with 1,369,751 parameters (see `LSTM_RNN_last_22.ipynb`)
13. Recorded diverse set of sEMG data (see `data/chris_final/`) with 4 new muscle groups - `2023/06/09`
14. Made training example sets with window sizes 5, 10, and 30.
15. Trained various models, experimented with hyperparameters and network architectures
16. Assessed data processing methods `2023/06/22`

### **Next Steps**
1. Build a real-time transcription app
2. Upgrade model to whispered/subvocal phoneme recognition.

## <a id="file-descriptions" style="color: inherit; text-decoration: none;"><u>File Descriptions</u></a>

- `Project_Methods.png` - image showing history of recording sessions 
- `git_push_script.bat` - batch script to quickly push all changes to the repository with a commit message
- `archive/` - folder which contains old model training and data processing notebooks
- `data/` - folder which contains the raw .csv files from the recordings, as well as formatted training example/label .npy files
- `gtp_convos/` - discussions with GPT-4 about the project
- `models/` - folder which contains saved models, not just weights
- `pictures/` - folder which contains pictures for the README.md
- `reports/` - folder which contains reports on Python notebooks, markdown reports of data processing and machine learning results and analysis

## <a id="file-descriptions" style="color: inherit; text-decoration: none;"><u>Results</u></a>

TBA - Need to do lots of stuff and put a description here