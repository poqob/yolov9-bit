import pandas as pd
import os

# Varsayım: Bu script, model_comparison_table.csv dosyasının bulunduğu dizinde çalıştırılır.
CSV_PATH = os.path.join(os.path.dirname(__file__), 'model_comparison_table.csv')

# Çıktı dosyası
BEST_MODELS_CSV = os.path.join(os.path.dirname(__file__), 'best_models.csv')
BEST_MODELS_MD = os.path.join(os.path.dirname(__file__), 'best_models.md')

# CSV dosyasını oku
df = pd.read_csv(CSV_PATH)

# Her mimari için en iyi modeli (en yüksek mAP@0.5) bul
best_by_arch = df.loc[df.groupby('architecture')['best_map50'].idxmax()]

# Tüm modeller arasında en iyi modeli bul (en yüksek mAP@0.5)
best_overall = df.loc[df['best_map50'].idxmax()]

# Sonuçları kaydet
best_by_arch.to_csv(BEST_MODELS_CSV, index=False)

# Markdown tablosu oluştur
md = '# En İyi Modeller Tablosu\n\n'
md += '| Model Mimarisi | Aktivasyon | Optimizasyon | Model Boyutu (MB) | mAP@0.5 | mAP@0.5:0.95 | Kesinlik | Hassasiyet | En İyi Epok | Toplam Epok | Box Loss | Cls Loss | DFL Loss |\n'
md += '|---------------|------------|--------------|-------------------|---------|--------------|----------|------------|------------|-------------|----------|----------|----------|\n'
for _, row in best_by_arch.iterrows():
    md += f"| {row['architecture']} | {row['activation']} | {row['optimizer']} | {row['model_size_mb']:.2f} | {row['best_map50']:.4f} | {row['best_map50_95']:.4f} | {row['best_precision']:.4f} | {row['best_recall']:.4f} | {int(row['best_epoch'])} | {int(row['epochs'])} | {row['last_box_loss']:.4f} | {row['last_cls_loss']:.4f} | {row['last_dfl_loss']:.4f} |\n"
md += '\n'
md += '## Tüm Modeller Arasında En İyi Model\n\n'
md += '| Model Mimarisi | Aktivasyon | Optimizasyon | Model Boyutu (MB) | mAP@0.5 | mAP@0.5:0.95 | Kesinlik | Hassasiyet | En İyi Epok | Toplam Epok | Box Loss | Cls Loss | DFL Loss |\n'
md += '|---------------|------------|--------------|-------------------|---------|--------------|----------|------------|------------|-------------|----------|----------|----------|\n'
md += f"| {best_overall['architecture']} | {best_overall['activation']} | {best_overall['optimizer']} | {best_overall['model_size_mb']:.2f} | {best_overall['best_map50']:.4f} | {best_overall['best_map50_95']:.4f} | {best_overall['best_precision']:.4f} | {best_overall['best_recall']:.4f} | {int(best_overall['best_epoch'])} | {int(best_overall['epochs'])} | {best_overall['last_box_loss']:.4f} | {best_overall['last_cls_loss']:.4f} | {best_overall['last_dfl_loss']:.4f} |\n"

with open(BEST_MODELS_MD, 'w') as f:
    f.write(md)

print(f"Her mimari için en iyi modeller {BEST_MODELS_CSV} ve {BEST_MODELS_MD} dosyalarına kaydedildi.")
print("Tüm modeller arasında en iyi model:")
print(best_overall)
