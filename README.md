# Bird Species Classification from Audio using Deep Learning 🎶🕊️

## 📌 Project Overview
This project investigates the use of **neural networks** to classify bird species from their sound recordings.  
By converting bird calls into **spectrograms** and applying deep learning techniques, the model learns to identify bird species automatically.  

The study focuses on two approaches:
- **Binary Classification**: Distinguish between two species (Northern Flicker vs. Blue Jay).
- **Multiclass Classification**: Classify 12 bird species from the dataset.

---

## 📊 Dataset
- Source: [Xeno-Canto](https://www.xeno-canto.org/) – a crowd-sourced bird sounds archive.  
- Pre-processed dataset of **12 bird species**, with **10 audio clips per species**.  
- Converted into **spectrogram images** (time-frequency representation of audio signals).  

Species included:  
`American Crow, Barn Swallow, Black-capped Chickadee, Blue Jay, Dark-eyed Junco, House Finch, Mallard, Northern Flicker, Red-winged Blackbird, Steller’s Jay, Western Meadowlark, White-crowned Sparrow`

---

## ⚙️ Tech Stack
- **Programming Language:** Python  
- **Libraries & Tools:** NumPy, Pandas, Matplotlib, Seaborn, Librosa, scikit-learn, TensorFlow/Keras  
- **Models:** Fully Connected Neural Networks (Dense layers), Sigmoid for binary classification, Softmax for multiclass classification  
- **Data Processing:** HDF5 storage, spectrogram extraction with Librosa, one-hot encoding  

---

## 🚀 Methodology
1. **Data Preprocessing**
   - Converted audio clips into **mel-spectrograms**.
   - Normalized and reshaped for neural network input.  

2. **Model Design**
   - **Binary Model:** Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(1, Sigmoid).  
   - **Multiclass Model:** Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(12, Softmax).  

3. **Training**
   - Optimizer: RMSProp  
   - Loss: Binary Crossentropy (binary), Categorical Crossentropy (multiclass)  
   - Evaluated different **batch sizes** and **epochs** to optimize accuracy.  

4. **Testing on External Audio**
   - Created spectrograms from raw `.mp3` clips.  
   - Applied trained models to classify unseen bird calls.  

---

## 📈 Results

### Binary Classification (Northern Flicker vs. Blue Jay)
- Best accuracy: **86.36%** (Batch size = 128, 15 epochs).  
- Training time: ~60s.  
- Model showed slight **overfitting** but strong ability to separate the two classes.  

### Multiclass Classification (12 species)
- Best accuracy: **66.38%** (Batch size = 128, 10 epochs).  
- Accuracy decreased as complexity increased.  
- Struggled to distinguish **Blue Jay vs. Western Meadowlark**.  

### External Test Clips
- On 3 new audio recordings, the model consistently predicted **American Crow** with high confidence.  
- Indicates robustness for that species, but highlights limitations in handling mixed/ambiguous calls.  

---

## 📌 Key Learnings
- Neural networks can effectively classify species from sound using spectrograms.  
- **Binary models** perform better due to reduced complexity.  
- **Multiclass classification** requires further refinement (e.g., CNNs, data augmentation).  
- Overfitting remains a challenge, pointing to the need for **regularization** or **larger datasets**.  

---

## 🔮 Future Improvements
- Use **Convolutional Neural Networks (CNNs)** for better feature extraction.  
- Apply **data augmentation** to expand dataset diversity.  
- Explore **RNNs/LSTMs** for sequential audio features.  
- Evaluate ensemble approaches or transfer learning.  

---

---

## ✨ Results in a Nutshell
- **Binary Model:** ~86% accuracy  
- **Multiclass Model:** ~66% accuracy  
- **External Testing:** Correctly identified American Crow  

This project demonstrates the potential of deep learning in **bioacoustics** and **wildlife monitoring**, providing an automated approach to bird species identification.  


