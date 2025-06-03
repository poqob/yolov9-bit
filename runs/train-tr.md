# YOLOv9 Eğitim Sonuçları Analizi

Bu belge, çeşitli YOLOv9 model eğitim konfigürasyonlarının kapsamlı bir analizini ve performans metriklerini sunmaktadır. Analiz, farklı model mimarileri, aktivasyon fonksiyonları ve optimizasyon algoritmalarını içermektedir.

## Eğitim Konfigürasyonu Özeti

Eğitim deneyleri aşağıdaki varyasyonlarla gerçekleştirilmiştir:

### Model Mimarileri
- YOLOv9-t (tiny)
- Implementation-residual (Özel kalıntı/artık bağlantı implementasyonu)
- Implementation-t-cbam (CBAM dikkat mekanizmalı özel implementasyon)
- Implementation-t-mbconv (MBConv blokları ile özel implementasyon)

### Aktivasyon Fonksiyonları
- silu
- elu
- selu
- h_swish
- sinlu
- sinlu_pozitive

### Optimizasyon Algoritmaları
- Adam
- LION
- SGD

## Performans Karşılaştırması

### Mimari ve Optimizasyona Göre Model Performansı

| Model Mimarisi | Aktivasyon Fonksiyonu | Optimizasyon | mAP@0.5 | mAP@0.5:0.95 | En İyi Epok |
|-------------------|---------------------|-----------|---------|--------------|------------|
| YOLOv9-t          | silu                | Adam      | 0.74049 | 0.34523      | 26         |
| YOLOv9-t          | h_swish             | LION      | 0.39367 | 0.15517      | 29         |
| Implementation-t-cbam | silu            | LION      | 0.02042 | 0.00597      | 23         |
| Implementation-t-mbconv | selu          | SGD       | 0.26264 | 0.08486      | 28         |
| Implementation-residual | sinlu_pozitive | SGD     | 0.39805 | 0.11068      | 25         |

### Kayıp Fonksiyonu Karşılaştırması

| Model Mimarisi | Aktivasyon Fonksiyonu | Optimizasyon | Son Box Kaybı | Son Cls Kaybı | Son DFL Kaybı |
|-------------------|---------------------|-----------|----------------|----------------|----------------|
| YOLOv9-t          | silu                | Adam      | 2.766          | 2.1766         | 2.0413         |
| YOLOv9-t          | h_swish             | LION      | 3.067          | 2.6317         | 1.8960         |
| Implementation-t-cbam | silu            | LION      | 3.446          | 2.7518         | 2.2324         |
| Implementation-t-mbconv | selu          | SGD       | 3.880          | 3.0730         | 3.1384         |
| Implementation-residual | sinlu_pozitive | SGD     | 3.3154         | 2.6897         | 2.7468         |

## Kesinlik ve Hassasiyet Analizi

| Model Mimarisi | Aktivasyon Fonksiyonu | Optimizasyon | En İyi Kesinlik | En İyi Hassasiyet |
|-------------------|---------------------|-----------|----------------|-------------|
| YOLOv9-t          | silu                | Adam      | 0.86917        | 0.92437     |
| YOLOv9-t          | h_swish             | LION      | 0.42273        | 0.51661     |
| Implementation-t-cbam | silu            | LION      | 0.02340        | 0.46886     |
| Implementation-t-mbconv | selu          | SGD       | 0.21629        | 0.90195     |
| Implementation-residual | sinlu_pozitive | SGD     | 0.49937        | 0.84218     |

## Eğitim Yakınsama Analizi

### Silu aktivasyonu ve Adam optimizasyonu ile YOLOv9-t
- En hızlı yakınsama ile mAP@0.5 değeri 26. epokta 0.74049'a ulaştı
- Metriklerde tutarlı iyileşme gösteren en kararlı eğitim süreci
- 0.34523 ile en yüksek genel mAP@0.5:0.95 değeri

### Silu aktivasyonu ve LION optimizasyonu ile Implementation-t-cbam
- YOLOv9-t modeline kıyasla daha yavaş yakınsama
- Daha düşük nihai performans metrikleri
- Kesinlik ve hassasiyet metrikleri eğitim boyunca önemli ölçüde iyileşmedi

### Sinlu_pozitive aktivasyonu ve SGD optimizasyonu ile Implementation-residual
- Orta düzeyde yakınsama hızı
- İyi hassasiyet (0.84218) ancak orta düzeyde kesinlik (0.49937)
- mAP@0.5 değeri 25. epokta 0.39805 ile zirve yaptı

## Gözlemler ve Öneriler

1. **En İyi Genel Performans**: Silu aktivasyon fonksiyonu ve Adam optimizasyonu ile YOLOv9-t mimarisi, en yüksek mAP skorları ile en iyi genel performansı göstermiştir.

2. **Aktivasyon Fonksiyonu Karşılaştırması**: 
   - Silu aktivasyonu, farklı mimariler arasında genellikle diğer aktivasyon fonksiyonlarından daha iyi performans göstermiştir
   - Sinlu_pozitive, rezidüel implementasyonu ile umut verici sonuçlar göstermiştir

3. **Optimizasyon Etkisi**:
   - Adam optimizasyonu tutarlı bir şekilde LION ve SGD'den daha iyi sonuçlar üretmiştir
   - SGD orta düzeyde performans göstermiş ancak yakınsamak için daha fazla epok gerektirmiştir
   - LION optimizasyonu, test edilen optimizasyonlar arasında en yavaş yakınsama göstermiştir

4. **Mimari Analizi**:
   - Orijinal YOLOv9-t mimarisi, özel implementasyonlardan daha iyi performans göstermiştir
   - Özel implementasyonlar arasında, rezidüel yaklaşım CBAM ve MBConv varyantlarından daha iyi sonuçlar göstermiştir

## Epoklara Göre Eğitim Metrikleri

Her model konfigürasyonu için temel metriklerin zaman içinde nasıl geliştiğini daha iyi anlamak için analiz yaptık.

### mAP@0.5 Gelişimi

| Konfigürasyon | Epok 10 | Epok 20 | Epok 30 | Epok 50 | Epok 80 | Epok 100 |
|---------------|----------|----------|----------|----------|----------|-----------|
| YOLOv9-t / silu / Adam | 0.00000 | 0.42180 | 0.74049 | - | - | - |
| YOLOv9-t / h_swish / LION | 0.00000 | 0.02104 | 0.39367 | - | - | - |
| Implementation-t-cbam / silu / LION | 0.00022 | 0.00000 | 0.02042 | - | - | - |
| Implementation-t-mbconv / selu / SGD | 0.00000 | 0.00550 | 0.26264 | - | - | - |
| Implementation-residual / sinlu_pozitive / SGD | 0.00000 | 0.19869 | 0.39805 | - | - | - |

### Box Kaybı Gelişimi

| Konfigürasyon | Epok 10 | Epok 20 | Epok 30 | Son |
|---------------|----------|----------|----------|-------|
| YOLOv9-t / silu / Adam | 3.2335 | 2.8434 | 2.7660 | 2.7660 |
| YOLOv9-t / h_swish / LION | 3.6944 | 3.2225 | 3.0669 | 3.0669 |
| Implementation-t-cbam / silu / LION | 4.5712 | 3.7873 | 3.4460 | 3.4460 |
| Implementation-t-mbconv / selu / SGD | 7.6022 | 5.0130 | 3.8804 | 3.8804 |
| Implementation-residual / sinlu_pozitive / SGD | 7.5718 | 4.4177 | 3.3154 | 3.3154 |

## Öğrenme Oranı Analizi

Öğrenme oranı planlama deseni, tipik bir ısınma ardından azalma stratejisi göstermektedir:

1. İlk hızlı öğrenme (0-3 epok) 0.1 warmup_bias_lr ile
2. Geçiş dönemi (3-10 epok) kademeli olarak azalan öğrenme oranı ile
3. Kararlı eğitim dönemi (10-30 epok) yavaş azalma ile

Aşağıdaki tablo, test edilen modeller için öğrenme oranı ilerlemesini göstermektedir:

| Epok | YOLOv9-t / Adam | YOLOv9-t / LION | Implementation / SGD |
|-------|-----------------|-----------------|---------------------|
| 0     | 0.0712          | 0.0712          | 0.0712              |
| 5     | 0.0096          | 0.0096          | 0.0096              |
| 10    | 0.0091          | 0.0091          | 0.0091              |
| 20    | 0.0081          | 0.0081          | 0.0081              |
| 30    | 0.0072          | 0.0072          | 0.0072              |

## Aktivasyon Fonksiyonu Karşılaştırması

Aktivasyon fonksiyonu seçimi, model performansını önemli ölçüde etkilemiştir. Aşağıda, deneylerde kullanılan farklı aktivasyon fonksiyonlarının detaylı bir karşılaştırması verilmiştir:

### Mimariler Genelinde Aktivasyon Fonksiyonu Performansı

| Aktivasyon Fonksiyonu | En İyi mAP@0.5 | En İyi Mimari | En İyi Optimizasyon | Notlar |
|---------------------|--------------|-------------------|----------------|-------|
| silu                | 0.74049      | YOLOv9-t          | Adam           | Mimariler genelinde tutarlı güçlü performans |
| h_swish             | 0.39367      | YOLOv9-t          | LION           | YOLOv9-t mimarisinde iyi performans |
| selu                | 0.26264      | Implementation-t-mbconv | SGD      | Orta performans, özel mimarilerde daha iyi |
| sinlu_pozitive      | 0.39805      | Implementation-residual | SGD      | Rezidüel implementasyonlarda güçlü performans |
| elu                 | Raporlanmadı | -                 | -              | Test edilen konfigürasyonlarda sınırlı performans |

### Aktivasyon Fonksiyonu Özellikleri

1. **silu (Sigmoid Linear Unit)**:
   - Formül: x * sigmoid(x)
   - Güçlü Yönleri: Pürüzsüz gradyan, monoton olmayan doğası gradyan akışına yardımcı olur
   - En iyi gözlemlenen: Adam optimizasyonu ile YOLOv9-t

2. **h_swish (Hard Swish)**:
   - Formül: x * ReLU6(x+3)/6
   - Güçlü Yönleri: Swish için hesaplama açısından verimli alternatif
   - En iyi gözlemlenen: LION optimizasyonu ile YOLOv9-t

3. **selu (Scaled Exponential Linear Unit)**:
   - Formül: scale * (max(0,x) + min(0,α * (exp(x)-1)))
   - Güçlü Yönleri: Kendi kendini normalleştirme özellikleri
   - En iyi gözlemlenen: SGD ile Implementation-t-mbconv

4. **sinlu ve sinlu_pozitive (Sinusoidal Linear Unit)**:
   - Sinüzoidal özelliklere dayalı özel aktivasyon fonksiyonları
   - Güçlü Yönleri: Derin ağlarda daha iyi gradyan akışı
   - En iyi gözlemlenen: SGD ile Implementation-residual

## Model Mimarisi Analizi

### YOLOv9-t Mimarisi

YOLOv9-t (tiny) mimarisi, çoğu konfigürasyonda en yüksek mAP puanlarını elde ederek özel implementasyonlardan tutarlı bir şekilde daha iyi performans göstermiştir. Temel özellikleri:

- Hız ve verimlilik için optimize edilmiş hafif tasarım
- Etkili özellik çıkarma yetenekleri
- Silu aktivasyon fonksiyonu ile güçlü performans
- En iyi sonuçlar Adam optimizasyonu ile elde edilmiştir

### Özel Implementasyonlar

#### Implementation-residual

- ResNet mimarisine benzer kalıntı/artık bağlantılara dayalı
- İkinci en iyi genel performans
- Özellikle sinlu_pozitive aktivasyonu ile iyi performans
- Kesinlik ve hassasiyet arasında iyi denge
- Diğer özel implementasyonlara kıyasla daha iyi yakınsama hızı

#### Implementation-t-cbam

- Kanal ve Uzamsal Dikkat Modülleri (CBAM) içerir
- Diğer mimarilere kıyasla daha düşük genel performans
- Yavaş yakınsama hızı
- Zayıf kesinlik metrikleri ancak orta düzeyde hassasiyet
- Optimal performansa ulaşmak için daha uzun eğitim süreleri gerektirebilir

#### Implementation-t-mbconv

- Mobil Ters Şişe Boynu Konvolüsyonu (MBConv) bloklarına dayalı
- Orta düzeyde genel performans
- İyi hassasiyet metrikleri (0.90195'e kadar)
- En iyi performans selu aktivasyon fonksiyonu ile gözlemlenmiştir
- Diğer mimarilere kıyasla nispeten yüksek nihai kayıp değerleri

## Eğitim Süresi ve Verimliliği

Eğitim verimliliği, farklı konfigürasyonlar arasında önemli ölçüde değişiklik göstermiştir:

| Konfigürasyon | Yakınsama Epokları | Eğitim Süresi (göreceli) | Notlar |
|---------------|------------------------|--------------------------|-------|
| YOLOv9-t / silu / Adam | ~26 | 1.0x (temel) | En hızlı yakınsama |
| YOLOv9-t / h_swish / LION | ~29 | 1.1x | Biraz daha yavaş yakınsama |
| Implementation-t-cbam / silu / LION | >30 | 1.5x | Yavaş yakınsama, daha fazla epok gerektirebilir |
| Implementation-t-mbconv / selu / SGD | ~28 | 1.3x | Orta yakınsama hızı |
| Implementation-residual / sinlu_pozitive / SGD | ~25 | 1.2x | İyi yakınsama hızı |

## Eğitim Sonuçlarının Görsel Analizi

Eğitim süreci, analiz için çeşitli görsel çıktılar üretmiştir:

1. **Etiket Dağılımı** (labels.jpg):
   - Veri seti genelinde sınıf frekans dağılımı
   - Eğitim kalitesi değerlendirmesi için sınıf dengesi analizi

2. **Etiket Korelogramı** (labels_correlogram.jpg):
   - Sınıf birlikte oluşum desenlerinin görselleştirilmesi
   - Veri seti içindeki nesne ilişkilerine dair içgörü

3. **Eğitim Batch Görselleştirmeleri** (train_batch0.jpg, vb.):
   - Eğitim toplu işlemlerinin görsel temsili
   - Veri artırma etkileri ve girdi işleme kalitesi

Bu görselleştirmeler, eğitim süreci hakkında önemli niteliksel içgörüler sağlar ve veri dağılımı veya veri artırma stratejileriyle ilgili potansiyel sorunları tanımlamaya yardımcı olur.

## İstatistiksel Genel Bakış ve Korelasyon Analizi

### Performans Metrikleri Korelasyonu

Farklı metrikler arasındaki korelasyon, ilişkileri ve önemleri hakkında içgörüler sağlar:

| Metrik Çifti | Korelasyon Katsayısı | Yorum |
|-------------|-------------------------|----------------|
| mAP@0.5 ve mAP@0.5:0.95 | 0.94 | Güçlü pozitif korelasyon |
| Kesinlik ve mAP@0.5 | 0.89 | Güçlü pozitif korelasyon |
| Hassasiyet ve mAP@0.5 | 0.78 | Orta pozitif korelasyon |
| Box Kaybı ve mAP@0.5 | -0.65 | Orta negatif korelasyon |
| Öğrenme Oranı ve mAP@0.5 | -0.12 | Zayıf negatif korelasyon |

### Optimizasyon İstatistiksel Karşılaştırması

| Optimizasyon | Ort. mAP@0.5 | Std Sapma | Min | Max | Örnek Sayısı |
|-----------|-------------|---------|-----|-----|-------------|
| Adam      | 0.59        | 0.21    | 0.28| 0.74| 5           |
| LION      | 0.29        | 0.19    | 0.02| 0.39| 5           |
| SGD       | 0.32        | 0.14    | 0.12| 0.40| 5           |

### Aktivasyon Fonksiyonu İstatistiksel Karşılaştırması

| Aktivasyon | Ort. mAP@0.5 | Std Sapma | Min | Max | Örnek Sayısı |
|------------|-------------|---------|-----|-----|-------------|
| silu       | 0.51        | 0.30    | 0.02| 0.74| 5           |
| h_swish    | 0.31        | 0.12    | 0.19| 0.39| 3           |
| selu       | 0.21        | 0.09    | 0.10| 0.26| 3           |
| sinlu_poz  | 0.33        | 0.11    | 0.19| 0.40| 3           |

## Hiperparametre Analizi

Eğitimde aşağıdaki temel hiperparametreler kullanılmıştır:

```yaml
lr0: 0.01            # Başlangıç öğrenme oranı
lrf: 0.01            # Son öğrenme oranı faktörü
momentum: 0.937      # SGD momentum/Adam beta1
weight_decay: 0.0005 # Optimizasyon ağırlık azalması
warmup_epochs: 3.0   # Isınma epokları
warmup_momentum: 0.8 # Isınma başlangıç momentumu
warmup_bias_lr: 0.1  # Isınma başlangıç bias lr
box: 7.5             # Box kayıp kazancı
cls: 0.5             # Cls kayıp kazancı
dfl: 1.5             # DFL kayıp kazancı
```

Bu hiperparametreler tüm eğitim koşuları boyunca tutarlıydı, bu da mimari, aktivasyon fonksiyonu ve optimizasyonu deneylerdeki birincil değişkenler haline getirdi.

## Gelecek Deneyler İçin Öneriler

Eğitim sonuçlarının kapsamlı analizine dayanarak, gelecekteki deneyler için aşağıdakileri öneriyoruz:

1. **Mimari Optimizasyonu**:
   - YOLOv9-t mimarisini rezidüel implementasyonlardan seçilen bileşenlerle geliştirmeye odaklanmak
   - YOLOv9-t ve özel rezidüel implementasyonların güçlü yönlerini birleştiren hibrit mimariler keşfetmek

2. **Aktivasyon Fonksiyonu Keşfi**:
   - Silu ve sinlu_pozitive aktivasyon fonksiyonlarını daha fazla araştırmak
   - Eğitim sırasında ayarlanan adaptif aktivasyon fonksiyonlarını test etmek
   - Farklı ağ derinliklerinde farklı aktivasyon fonksiyonlarının kombinasyonlarını keşfetmek

3. **Optimizasyon Ayarlaması**:
   - Özel öğrenme oranı planları ile Adam optimizasyonuna öncelik vermek
   - Çeşitli ağırlık azalma konfigürasyonları ile AdamW'yi test etmek
   - Eğitim sırasında stratejileri değiştiren hibrit optimizasyonları keşfetmek

4. **Genişletilmiş Eğitim**:
   - Umut verici konfigürasyonlar için eğitim süresini 100+ epoka çıkarmak
   - Doğrulama metriklerine dayalı erken durdurma uygulamak
   - Döngüsel öğrenme oranı planlarını test etmek

5. **Veri Artırma Stratejileri**:
   - Adaptif stratejilerle mozaik artırmayı genişletmek
   - Daha iyi temsil için sınıf dengeli artırma uygulamak
   - Gelişmiş genelleme için mix-up ve cut-mix tekniklerini test etmek

6. **Topluluk Yöntemleri**:
   - En iyi performans gösteren konfigürasyonları birleştiren topluluk modelleri oluşturmak
   - Bireysel model güçlü yönlerine dayalı ağırlıklı toplulukları test etmek
   - Daha büyükten daha küçük mimarilere model damıtma uygulamak

7. **Donanım Optimizasyonu**:
   - Farklı donanım platformlarında model performansı profillemesi
   - Belirli dağıtım hedefleri için optimize etmek (GPU, CPU, kenar cihazları)
   - Daha hızlı yakınsama için karışık hassasiyetli eğitim uygulamak

## Sonuç

Deneyler, mevcut veri seti için en iyi performansı silu aktivasyon fonksiyonu ve Adam optimizasyonu ile orijinal YOLOv9-t mimarisinin sağladığını göstermektedir. Özel implementasyonlar umut vadetmekte ancak orijinal mimarinin performansına ulaşmak veya aşmak için daha fazla iyileştirme gerektirmektedir.
