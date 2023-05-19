# NeuroTechSC - Machine Learning

This is the Git repository for the Machine Learning team of the NeuroTechSC organization. The main goal of our project is to detect subvocal phonemes from surface Electromyography (sEMG) data using a Deep Neural Network model implemented in Python. This repository provides the Python notebooks for the sEMG data preprocessing and various model trainings.

## General Research Abstract for Project

Subvocalization refers to the internal speech that occurs while reading or thinking without producing any audible sound. It is accompanied by the activation of the muscles involved in speech production, generating electrical signals known as electromyograms (EMGs). These signals can potentially be used for silent communication and assistive technologies, improving the lives of people with speech impairments and enabling new forms of human-computer interaction. However, the accurate recognition of subvocal EMGs remains a challenge due to the variability in signal patterns and electrode placement. Our proposed solution is to develop an advanced subvocal EMG recognition system using machine learning techniques. By analyzing the EMG signals, our system will be able to identify the intended speech content and convert it into text. This technology will be applicable in various fields, including silent communication for military or emergency personnel, assistive devices for people with speech impairments, and hands-free control of computers and other electronic devices.

## Machine Learning Plan Overview (Original)

This project aims to improve the performance of subvocal phoneme detection using machine learning techniques. Subvocal phonemes are the speech components generated when a person talks to themselves without producing any sound. Detecting these phonemes has various applications, including silent communication devices and assistive technologies for individuals with speech impairments.

The TMC-ViT model used in this repository is a novel deep learning architecture that leverages the benefits of vision transformers and temporal multi-channel features to achieve improved performance on sEMG data. This model outperforms conventional methods such as CNNs and LSTMs in subvocal phoneme detection tasks. 

## Machine Learning Plan Amendment

Now considering the use of a LSTM/RNN for phoneme recognition instead of the TMC-ViT, check `gtp_convos/gpt_convo_2.md` for more information.

## Project Timeline

1. Came up with a design plan and chose model architecture (TMC-ViT)
2. Collected data for 5 phonemes
3. Finished preprocessing data
4. Analyzed data and presented findings at the California Neurotech Conference on April 29th, 2023 (see Research Abstract section)
5. Assessed model viability and are considering a pivot from TMC-ViT to an LSTM/RNN network

### Next Steps

1. Collect data on more phonemes, possibly modify the Cyton OpenBCI board to support 8 muscle groups instead of 4
2. Make a pivot to RNN/LSTM or another architecture
3. Assess the viability of a second model to correct phoneme/letter-level errors
4. Build a real-time transcription app

## File Descriptions

- `LSTM_RNN.ipynb` - a model that reached ~100% test accuracy on our 5 phonemes, has training code + visualization
- `EMG_Data_Processing.ipynb` - preprocessing script that cleans the data and formats it into training examples
- `gtp_convos/gpt_convo.md` - a discussion on processing .adc files and modifying TMC-ViT code
- `gtp_convos/gpt_convo_2.md` - a discussion on using transformer and RNN architectures for subvocal phoneme prediction
- `data/` - folder which contains the raw .csv files from the recordings, as well as formatted example/label .npy files
