Deney ortamı Kurulumu (Experimental Setup) :
3 İnput Ham XYZ_Acc Verisi, Early_stopping_off, BiGRU, Epoch150_BatchSize32_nHidden128_TimeStep5_TimeSegment20

Deneysel Sonuçlar (Experimental Results)

                precision    recall  f1-score   support

 MERDİVEN İNME       0.93      0.92      0.93       178
        UZANMA       0.99      0.99      0.99       193
        OTURMA       0.99      0.99      0.99       176
  AYAKTA DURMA       0.99      0.99      0.99       175
MERDİVEN ÇIKMA       0.97      0.98      0.98       183
        YÜRÜME       0.96      0.97      0.96       179
       ZIPLAMA       0.94      0.92      0.93       175

      accuracy                           0.97      1259
     macro avg       0.97      0.97      0.97      1259
  weighted avg       0.97      0.97      0.97      1259

Roc_AUC= 0.9964036628589841
