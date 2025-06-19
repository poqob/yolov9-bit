# En İyi Modeller Tablosu

| Model Mimarisi | Aktivasyon | Optimizasyon | Model Boyutu (MB) | mAP@0.5 | mAP@0.5:0.95 | Kesinlik | Hassasiyet | En İyi Epok | Toplam Epok | Box Loss | Cls Loss | DFL Loss |
|---------------|------------|--------------|-------------------|---------|--------------|----------|------------|------------|-------------|----------|----------|----------|
| implementation-residual | elu | SGD | 6.59 | 0.8759 | 0.5961 | 0.8462 | 0.8843 | 81 | 100 | 1.6149 | 1.1922 | 1.5159 |
| implementation-residual_h | swish | SGD | 6.59 | 0.8759 | 0.5961 | 0.8462 | 0.8843 | 81 | 100 | 1.6149 | 1.1922 | 1.5159 |
| implementation-residual_sinlu | pozitive | SGD | 6.59 | 0.8759 | 0.5961 | 0.8462 | 0.8843 | 81 | 100 | 1.6149 | 1.1922 | 1.5159 |
| implementation-t-cbam | elu | SGD | 7.59 | 0.8875 | 0.6265 | 0.8363 | 0.8869 | 94 | 100 | 1.7613 | 1.3646 | 1.6064 |
| implementation-t-cbam_h | swish | SGD | 7.59 | 0.8875 | 0.6265 | 0.8363 | 0.8869 | 94 | 100 | 1.7613 | 1.3646 | 1.6064 |
| implementation-t-cbam_sinlu | pozitive | SGD | 7.59 | 0.8875 | 0.6265 | 0.8363 | 0.8869 | 94 | 100 | 1.7613 | 1.3646 | 1.6064 |
| implementation-t-mbconv | elu | SGD | 6.72 | 0.8681 | 0.5770 | 0.8165 | 0.8679 | 89 | 100 | 1.7037 | 1.2521 | 1.6062 |
| implementation-t-mbconv_h | swish | SGD | 6.72 | 0.8681 | 0.5770 | 0.8165 | 0.8679 | 89 | 100 | 1.7037 | 1.2521 | 1.6062 |
| implementation-t-mbconv_sinlu | pozitive | SGD | 6.72 | 0.8681 | 0.5770 | 0.8165 | 0.8679 | 89 | 100 | 1.7037 | 1.2521 | 1.6062 |
| yolov9-t | elu | SGD | 5.79 | 0.8683 | 0.5835 | 0.8518 | 0.8349 | 88 | 100 | 1.5713 | 1.1690 | 1.4949 |
| yolov9-t_h | swish | SGD | 5.79 | 0.8683 | 0.5835 | 0.8518 | 0.8349 | 88 | 100 | 1.5713 | 1.1690 | 1.4949 |
| yolov9-t_sinlu | pozitive | SGD | 5.79 | 0.8683 | 0.5835 | 0.8518 | 0.8349 | 88 | 100 | 1.5713 | 1.1690 | 1.4949 |

## Tüm Modeller Arasında En İyi Model

| Model Mimarisi | Aktivasyon | Optimizasyon | Model Boyutu (MB) | mAP@0.5 | mAP@0.5:0.95 | Kesinlik | Hassasiyet | En İyi Epok | Toplam Epok | Box Loss | Cls Loss | DFL Loss |
|---------------|------------|--------------|-------------------|---------|--------------|----------|------------|------------|-------------|----------|----------|----------|
| implementation-t-cbam | elu | SGD | 7.59 | 0.8875 | 0.6265 | 0.8363 | 0.8869 | 94 | 100 | 1.7613 | 1.3646 | 1.6064 |
