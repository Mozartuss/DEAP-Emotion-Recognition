# Emotion recognition with the LSTM RNN

Implement an modified version of the LSTM RNN Model wich is used by Acharya D. et al. [[1]](#1) to get an accuracy of about 92.17% (Arousal) and about 94.46% (Valence)

## Preprocessing

Here, the DEAP dataset [[2]](#2) is used, where each of the 32 participant's data consists of 8064 readings for 32 EEG
channels and for each of the 40 video trials.

    Shape: (Subjects, Trials, Steps, Channels)
    Shape: (32, 40, 8064, 32)

Using the FFT to extract the seperate Bandwaves Delta-δ (1–4 Hz), Theta-θ (4–8 Hz), Alpha-α (8–14 Hz), Beta-β (14–31 Hz), and Gamma-γ (31–50 Hz).
The FFT use a window of 256 which averages the band power of 2sec each video, wherby the Window slides every 0.125 sec.

            
    Shape: (Subjects, Trials, FFT steps, Channels, Bandwaves)
    Shape: (32, 40, 488, 32, 5)
    Transform to 
    Shape: (624640, 160, 1)

## LSTM architecture

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
bidirectional (Bidirectional (None, 160, 256)          133120    
_________________________________________________________________
dropout (Dropout)            (None, 160, 256)          0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 160, 256)          525312    
_________________________________________________________________
dropout_1 (Dropout)          (None, 160, 256)          0         
_________________________________________________________________
lstm_2 (LSTM)                (None, 160, 64)           82176     
_________________________________________________________________
dropout_2 (Dropout)          (None, 160, 64)           0         
_________________________________________________________________
lstm_3 (LSTM)                (None, 160, 64)           33024     
_________________________________________________________________
dropout_3 (Dropout)          (None, 160, 64)           0         
_________________________________________________________________
lstm_4 (LSTM)                (None, 32)                12416     
_________________________________________________________________
dropout_4 (Dropout)          (None, 32)                0         
_________________________________________________________________
dense (Dense)                (None, 16)                528       
_________________________________________________________________
activation (Activation)      (None, 16)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 34        
_________________________________________________________________
activation_1 (Activation)    (None, 2)                 0         
=================================================================
Total params: 786,610
Trainable params: 786,610
Non-trainable params: 0
_________________________________________________________________
```

## Classification
The DEAP dataset contains 4 different labels: dominance, liking, arousal, and valence. Here we used Arousal and Valence to obtain emotional trends in the Russell's circumplex model.
To predict trends only, we need to threshold the labels in the middle to obtain binary values, since each label in the DEAP dataset was scored between 1 and 10.

## Conclusion
After training the moddel with an 75/25 split we get an accuracy of about 92.17% and 0.2001 as Loss-value (Arousal) (Left) and about 94.46% and 0.1553 as Loss-value (Valence) (Right)


<p align="middle">
    <img src="https://user-images.githubusercontent.com/32893711/150337781-f1035566-19ce-4e00-9e6b-f523db706dfd.jpg" alt="Arousal" width="45%"/>
    <img src="https://user-images.githubusercontent.com/32893711/150751426-829e6310-7275-4400-bddc-86f6fda6cfd0.png" alt="Valence" width="45%"/>
</p>

In order to achieve a better result, the channel optimization algorithm Principal component analysis (PCA) was applied, however, this did not optimize the accuracy in classifying the arousal, also the training process was significantly less effective than in the previous attempt. Arousal was classified with an accuracy of 83.39% and a loss of 0.3658

<p align="middle">
    <img src="https://user-images.githubusercontent.com/32893711/151954460-e1ce13e9-efb6-4caf-b964-1ffa8da84785.jpg" alt="Arousal PCA" width="45%"/>
</p>

With the channel-selected algorithm Minimum redundancy maximum relevance (mrmr), the same accuracy was achieved as without channel optimisation, however, 20 channels were still used here, probably only when more channels are removed different results are obtained. Arousal was classified with an accuracy of 92.74% and a loss of 0.1892, and Valence with an Accuracy of 92.36% and a loss of 0.1983

<p align="middle">
    <img src="https://user-images.githubusercontent.com/32893711/152510682-09599d95-6810-4cf4-abae-54d49bcda247.png" alt="Arousal MRMR" width="45%"/>
    <img src="https://user-images.githubusercontent.com/32893711/152136440-59c6730c-1544-479f-bff1-1e6f350d3410.png" alt="Valence MRMR" width="45%"/>
</p>



## References

<a id="1">[1]</a>
Acharya, D., Jain, R., Panigrahi, S.S., Sahni, R., Jain, S., Deshmukh, S.P., Bhardwaj, A.: Multi-class Emotion Classification Using EEG Signals. In: Garg, D., Wong, K., Sarangapani, J., Gupta, S.K. (eds.) Advanced Computing. 10th International Conference, IACC 2020, Panaji, Goa, India, December 5–6, 2020, Revised Selected Papers, Part I. Springer eBook Collection, vol. 1367, pp. 474–491. Springer Singapore; Imprint Springer, Singapore (2021). doi: 10.1007/978-981-16-0401-0_38

<a id="2">[2]</a>
Koelstra, S., Muhl, C., Soleymani, M., Lee, J.-S., Yazdani, A., Ebrahimi, T., Pun, T., Nijholt, A., Patras, I.: DEAP: A
Database for Emotion Analysis ;Using Physi-ological Signals. IEEE Transactions on Affective Computing, vol. 3, 18–31 (
2012). doi: 10.1109/T-AFFC.2011.15

