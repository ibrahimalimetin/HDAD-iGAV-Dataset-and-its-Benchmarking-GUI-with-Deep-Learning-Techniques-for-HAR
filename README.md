# HDAD iGAV Dataset and its Benchmarking GUI with Deep Learning Techniques for HAR

İnsanların gün içerisinde gerçekleştirdikleri aktivitelerin sınıflandırılmasında kullanılmak üzere yeni bir veri kümesi sunduğumuz çalışmada; hem derin öğrenme modellerinin (Uzun-Kısa Süreli Bellek (UKSB), Çift Yönlü Uzun-Kısa Süreli Bellek (Çy-UKSB), Kapılı Tekrarlayan Birim (KTB), Çift Yönlü Kapılı Tekrarlayan Birim (Çy-KTB) ve Evrişimli Sinir Ağları (ESA)) hem de halkın kullanımına açık HAR (Human Activity Recognition, İnsan Aktivitesi Tanıma) veri setlerinin kıyaslaması gerçekleştirilerek başarım sonuçları verilmiştir. Executable dosyası olarak verilen uygulama üzerinden testlerinizi gerçekleştirebilir, ilgili linkler üzerinden de projenin detaylarını inceleyebilirsiniz. 

![1](https://user-images.githubusercontent.com/11638083/113777365-a5758200-9733-11eb-85f8-5c8faef80435.jpg)

Yukarıda ekran görüntüsü paylaşılmış olan yazılımsal araç bireylerin günlük aktivitelerinin sınıflandırma işleminde kullanıcıya seçim yapma imkânı sunmaktadır. Tasarlanan yazılımsal aracın özellikleri ve seçim yaptırdığı parametrelerin açıklamaları aşağıdaki gibidir:

Derin mimari model seçimi, eklenen bu özellik ile UKSB, KTB, Çift yönlü UKSB, Çift yönlü KTB ve 1B ESA için oluşturulan modellerin deneylerde kullanılmak için seçimi mümkün kılmaktadır. 
Veri kümesi seçimi, bu seçim bir önceki özellikte belirtilen deneylerin gerçekleştirilmesi için seçilen derin mimari modelinin UCI, WISDM ya da İGAV kümelerinden hangisi ile test edilmesinin gerçekleştirileceğine karar verilmesini sağlamaktadır.
Veri kümesi göster butonu, seçimi yapılan veri kümesinin değerlerinin görüntülenmesini sağlamaktadır
Eğitim yineleme sayısı (epoch), eğitilecek olan modelin bir kez gerçekleşen işlemde eğitilememesi sebebi ile tekrardan ağırlıklarının güncellenmesi gerekmektedir. Eğitimi gerçekleştirilen her yeni verinin tekrardan ağırlık hesabı yapılır. Bu da modelin en uygun ağırlıklarını hesaplamasını sağlamaktadır. Gerçekleştirilen tüm bu adımları sayısı eğitim yineleme sayısını ifade etmektedir. Tasarlanan projede bu değerin seçilmesi mümkün kılınmıştır.
Katman sayısı, seçilen modelin kaç katmanda işlem yapacağının belirlenmesini sağlamaktadır.
Düğüm sayısı, seçilen katman sayısında kaç adet nöron olacağının belirtilmesini sağlamaktadır.
Erken durdurma (early stopping), veri kümesi eğitimini gerçekleştirirken, eğitim hata oranının sürekli azaldığı görülmektedir; fakat eğitim durdurulmaz ise hata oranı yükselmeye başlar ve bir süre sonra aşırı öğrenme meydana gelmektedir. Eğitim sırasında modelin az yineleme sayısı ile öğreneceği verileri öğrenememesi ile çok yüksek yineleme sayısında da aşırı öğrenme durumunu düşmemesi için erken durdurma yöntemi kullanılmaktadır. Bu özellik tasarlanan yazılımsal araçta belirtilen eğitim yineleme sayısından az bir yinelemede öğrenme oranı yüksek ise deneyin durdurulmasını sağlamaktadır.
Parçalanma boyutu (batch size), veri kümesinin parçalar halinde işleme alınmasını için belirlenen sayıyı ifade etmektedirler. 
Zaman bölütü (time segment), birim saniyede geçen örnekleme oranını belirtmektedir. Örneğin UCI veri kümesinde, 50 Hz frekans (2.56 sn ’de bir örnekleme) ile 2.56 sn ’lik pencere tercih edilmiştir. Bu da 2.56 sn ’de 128 örnek ile eğitim yapıldığını ifade etmektedir.
Zaman adımı (time step), ise her bir pencere başına yapılan okuma miktarını ifade etmektedir. Örneğin, WISDM veri kümesinde 20 Hz frekans (50 ms ’de bir örnekleme) ile her bir pencerede 10 okuma gerçekleştirilmiştir.

İGAV kümesinde veri edinimi (data acquisition), IOS akıllı cep telefonunun ivmeölçer ve jiroskop duyargaları ile 4 dinamik ve 3 statik aktiviteyi her bir aktivitenin iki farklı pozisyonda toplam 15 sn gerçekleştirilmesi ile sağlanmıştır. Bu aktiviteler farklı ortam koşullarında 25 - 55 yaş aralığındaki 5 erkek ve 5 kadın olmak üzere toplamda 10 gönüllünün bellerine yerleştirilerek gerçek zamanlı toplanmıştır. Aktivite verileri 20Hz örnekleme ile alınmıştır. Her bir aktivite için 15 sn uzunluğunda bir hareket verisi kaydedilmişti. Bu durumda her bir pencerelenmiş aktivite örneği için 300 adet veri demektir. 10 gönüllüye ait, 7 farklı aktivite örneği çıkarılarak toplam örnek veri sayısı 21000 olarak hesaplanmıştır. Çalışmamızda, pencere başına 20 okuma yapılması tercih edilmiştir.


1 BOYUTLU EVRİŞİMLİ SİNİR AĞININ WISDM VERİ KÜMESİ ÜZERİNDE ERKEN DURDURMA AÇIK OLUP, EĞİTİM YİNELEME SAYISI 27. ADIMDA TAMAMLANMIŞ GİRDİ VERİSİNİN PARÇALANMA BÜYÜKLÜĞÜ 32 OLAN TESTİNE İLİŞKİN MODEL DOĞRULUĞU, MODEL KAYBI, KARMAŞIKLIK MATRİSİ VE ROC EĞRİ GRAFİKLERİ AŞAĞIDAKİ GİBİDİR:

![image](https://user-images.githubusercontent.com/11638083/113777477-cf2ea900-9733-11eb-87c7-1d7edc83fe91.png)

![image](https://user-images.githubusercontent.com/11638083/113777496-d3f35d00-9733-11eb-9d5f-43a7d6ee6662.png)

![image](https://user-images.githubusercontent.com/11638083/113777511-d8b81100-9733-11eb-941e-076528f4f1ea.png)

![image](https://user-images.githubusercontent.com/11638083/113777517-de155b80-9733-11eb-92b5-860af1886899.png)


Kaynak:

[1] Metin, İ , Karasulu, B . (2021). İnsanın günlük aktivitelerinin yeni bir veri kümesi: Derin öğrenme tekniklerini kullanarak sınıflandırma performansı için kıyaslama sonuçları . Gazi Üniversitesi Mühendislik Mimarlık Fakültesi Dergisi , 36 (2) , 759-778 . DOI: https://doi.org/10.17341/gazimmfd.772849

[2] Guillaume Chevalier, LSTMs for Human Activity Recognition, 2016, https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition

[3] @article{DBLP:journals/corr/abs-1708-08989, author = {Yu Zhao and Rennong Yang and Guillaume Chevalier and Maoguo Gong}, title = {Deep Residual Bidir-LSTM for Human Activity Recognition Using Wearable Sensors}, journal = {CoRR}, volume = {abs/1708.08989}, year = {2017}, url = {http://arxiv.org/abs/1708.08989}, archivePrefix = {arXiv}, eprint = {1708.08989}, timestamp = {Mon, 13 Aug 2018 16:46:48 +0200}, biburl = {https://dblp.org/rec/bib/journals/corr/abs-1708-08989}, bibsource = {dblp computer science bibliography, https://dblp.org} }

[4] WISDM dataset website. WISDM Lab Dataset. https://www.cis.fordham.edu/wisdm/dataset.php

[5] UCI dataset website. Human activity recognition using smartphones dataset. https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
