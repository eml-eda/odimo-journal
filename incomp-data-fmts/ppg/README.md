# Preliminary results on the PPG DaLiA task

The PPG DaLiA task deals with heart rate (HR) estimation using data from a photoplethysmographic (PPG) sensor and from a tri-axial accelerometer to compensate motion artifacts.
In particular, this dataset contains PPG and 3D accelerometer data collected during daily life activities from 15 subjects, with ground-truth heart-rate measures performed using ECG.
The traditional training and evaluation scheme for this dataset is based on the Leave-One-Subject-Out (LOSO) cross-validation scheme, where each fold is composed of one subject used as the test set, four subjects used as validation, and the remaining ones as the training set.
The task performance of the algorithm, given it is a regression task, is measured as the Mean Absolute Error (MAE) on the predicted HR, in Beats Per Minute (BPM); thus, a lower value is better.

As a reference and seed network for these experiments, we considered a state-of-the-art Temporal Convolutional Network (TCN), composed of 1D convolutions with increasing dilation over layers, called [TEMPONet](https://ieeexplore.ieee.org/abstract/document/8930945). 
TEMPONet has been specifically designed for time series processing. 
Thus, overall, this experiment shows ODiMO's generality from three different angles: 
  - (i) we target a different application domain from computer vision, 
  - (ii) we consider a different type of DNN model (TCNs),
  - (iii) a different type of machine learning task, i.e., a regression with a continuous output.

Please note that these results are preliminary, indeed, we evaluated the ODiMO approach on two randomly selected subjects (Subjects 3 and 9).

![Preliminary Results](/odimo-journal/incomp-data-fmts/ppg/diana_ppg_dalia.png)
