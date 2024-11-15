# Bird-species-classification-using-neural-network-models

# Introduction
This project classifies bird species by analyzing audio recordings using neural networks. It leverages both binary and multiclass models trained on time-frequency representations (spectrograms) of bird sounds.

# Dataset
Source: Xeno-Canto bird sounds archive.
Preprocessed Data: 12 bird species with 10 audio clips each, converted to spectrograms.
# Models and Architecture
# Binary Classification Model: Classifies between Northern Flicker and Blue Jay with 86.36% accuracy.
# Multiclass Classification Model: Identifies all 12 bird species with an accuracy of 66.38%.
# Neural Network Layers:
Binary Model: Flatten, Dense layers (128 and 64 neurons), Sigmoid activation.
Multiclass Model: Similar structure with a Softmax layer for multiple class outputs.
Key Features
Data Preprocessing:
Spectrogram generation from audio files.
Train-test split with one-hot encoding for multiclass.
Model Training:
Binary model for two species, multiclass model for 12 species.
Hyperparameter tuning for batch sizes and epochs.
Evaluation and Validation:
Accuracy metrics and loss curves for model performance analysis.
Confusion matrix for detailed prediction insights.
Results
Binary Model: 86.36% accuracy
Multiclass Model: 66.38% accuracy
External Validation: Successfully identified American Crow in test clips.
Installation and Setup
Clone this repository.
Install dependencies: TensorFlow, Keras, Librosa, Matplotlib.
Run preprocess_audio.py to convert audio files to spectrograms.
Use train_binary_model.py and train_multiclass_model.py to train models.
Evaluate using external test audio files.
Usage
Preprocess Audio Files: Converts audio clips into spectrogram images.
Train Models: Choose binary or multiclass scripts for training.
Validate Results: Run predictions on external audio files.
Future Improvements
Use larger datasets and additional bird species.
Test CNNs or other architectures for enhanced accuracy.
Explore fine-tuning on more varied environmental sounds.
Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss proposed updates.
