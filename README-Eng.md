# HDAD iGAV Dataset and its Benchmarking GUI with Deep Learning Techniques for HAR

In the study we present a new data set to be used in classifying the activities people perform during the day; A comparative study was perforned to show the use of some deep learning architectural models on the publicly available datasets, and also their performance evaluation results are given (Long-Short Term Memory, bi-directional Long-Short Term Memory, Gated Recurrent Unit, bi-directional Gated Recurrent Unit and Convolutional Neural Networks).

![image](https://user-images.githubusercontent.com/11638083/113778194-c8ecfc80-9734-11eb-8b63-55b99a388e71.png)

The software tool, whose screenshot was shared above, offers the user the opportunity to choose in the classification process of the daily activities of individuals. The features of the designed software tool and the explanations of the parameters chosen are as follows:
Deep architectural model selection, with this added feature, enables the selection of models created for LSTM, GRU, Bi-directional LSTM, Bi-directional GRU and 1D CNN for use in experiments.
Data set selection (in Turkish: Veri kümesi seçimi), this selection enables to decide whether to test the deep architectural model selected to perform the experiments mentioned in the previous feature with UCI, WISDM or HDAD clusters.
The Show data set button (in Turkish: Veri kümesi göster butonu) allows the values of the selected data set to be displayed.
The number of training epoch requires updating the weights of the model to be trained since it cannot be trained in one operation. Weight calculation is made for each new data that is trained. This allows the model to calculate the most appropriate weights. The number of all these steps performed represents the number of training epoch. It has been made possible to select this value in the designed project.
The number of layers determines how many layers the selected model will process.
The number of nodes allows it to indicate how many neurons will be in the selected number of layers.
While early stopping performs dataset training, it is observed that the training error rate is constantly decreasing; However, if the training is not stopped, the error rate starts to increase and after a while, excessive learning occurs. During the training, the early stop method is used to prevent the over-learning status from decreasing in very high number of epoch, as the model cannot learn the data it will learn with a low number of epoch. This feature ensures that the experiment is stopped if the learning rate is high in less than the number of training epoch specified in the designed software tool.
Batch size refers to the number of times the data set is processed in chunks.
Time segment (in Turkish: Zaman bölütü) refers to the rate of sampling per second. For example, in the UCI data set, a 2.56 sec window with a 50 Hz frequency (one sampling at 2.56 sec) was preferred. This means that training was carried out with 128 samples in 2.56 seconds.
Time step (in Turkish: Zaman adımı) refers to the amount of reading done per window. For example, 10 readings were performed in each window with a frequency of 20 Hz (one sampling at 50 ms) in the WISDM dataset.


Data acquisition in the HDAD (IGAV) was achieved by performing 4 dynamic and 3 static activities with the accelerometer and gyroscope sensors of the IOS smart phone in two different positions for a total of 15 seconds. These activities were collected in real time by placing them on the waist of a total of 10 volunteers, 5 males and 5 females, between the ages of 25 and 55, in different environmental conditions. Activity data were taken at 20Hz sampling. A 15-second long motion data was recorded for each activity. This means 300 data for each windowed activity instance. By extracting 7 different activity samples from 10 volunteers, the total number of sample data was calculated as 21000. In our study, 20 readings per window were preferred.

EARLY STOP ON 1D CNN WISDM DATA SET IS ON, THE NUMBER OF TRAINING EPOCH COMPLETED IN STEP 27. SIZE OF INPUT BATCH SIZE OF THE INPUT DATA IS 32 MODEL ACCURACY RELATED TO THE TEST OF THE MATERIAL 

![image](https://user-images.githubusercontent.com/11638083/113778256-dd30f980-9734-11eb-87f1-bb8db89e1540.png)

![image](https://user-images.githubusercontent.com/11638083/113778267-e15d1700-9734-11eb-915c-51d9d1b934a2.png)

![image](https://user-images.githubusercontent.com/11638083/113778276-e4f09e00-9734-11eb-8d35-0c62403b1326.png)

![image](https://user-images.githubusercontent.com/11638083/113778283-e7eb8e80-9734-11eb-8c74-d57cc1a74225.png)

CITATION

[1] Metin, İ , Karasulu, B . (2021). İnsanın günlük aktivitelerinin yeni bir veri kümesi: Derin öğrenme tekniklerini kullanarak sınıflandırma performansı için kıyaslama sonuçları . Gazi Üniversitesi Mühendislik Mimarlık Fakültesi Dergisi , 36 (2) , 759-778 . DOI: https://doi.org/10.17341/gazimmfd.772849

[2] Guillaume Chevalier, LSTMs for Human Activity Recognition, 2016, https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition

[3] @article{DBLP:journals/corr/abs-1708-08989,
  author    = {Yu Zhao and
               Rennong Yang and
               Guillaume Chevalier and
               Maoguo Gong},
  title     = {Deep Residual Bidir-LSTM for Human Activity Recognition Using Wearable
               Sensors},
  journal   = {CoRR},
  volume    = {abs/1708.08989},
  year      = {2017},
  url       = {http://arxiv.org/abs/1708.08989},
  archivePrefix = {arXiv},
  eprint    = {1708.08989},
  timestamp = {Mon, 13 Aug 2018 16:46:48 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1708-08989},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

[4] WISDM veri kümesi websitesi. WISDM Lab Dataset.
https://www.cis.fordham.edu/wisdm/dataset.php

[5] UCI veri kümesi websitesi. Human activity recognition using smartphones dataset. 
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

[6] Venelin Valkov, (2017), Medium Website (Human Activity Recognition using LSTMs on Android): 
https://medium.com/@curiousily/human-activity-recognition-using-lstms-on-android-tensorflow-for-hackers-part-vi-492da5adef64

[7] Tomasz Bartkowiak, (2021), Github Website (Human Activity Recognition on the Wireless Sensor Data Mining (WISDM) dataset using Bidirectional LSTM Recurrent Neural Networks):  
https://github.com/bartkowiaktomasz
