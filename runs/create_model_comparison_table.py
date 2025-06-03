#!/usr/bin/env python3
import os
import pandas as pd
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Eğitim dizinini tanımla - projenin kök dizinine göre relatif yol kullan
SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_PATH))
TRAIN_DIR = os.path.join(PROJECT_ROOT, "runs", "train")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "runs", "analysis")

# Çıktı dizini yoksa oluştur
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_model_info(model_dir):
    """Model dizin adından mimari, aktivasyon ve optimizasyon bilgilerini çıkarır."""
    try:
        # Model adını parçala (mimari_aktivasyon_optimizasyon)
        parts = model_dir.split('_')
        if len(parts) < 3:
            print(f"Uyarı: {model_dir} model adı beklenen format ile uyuşmuyor (mimari_aktivasyon_optimizasyon)")
            return None, None, None
        
        # Son parça optimizasyon olabilir ancak sayı içeriyorsa temizle
        optimizer = re.sub(r'[0-9]', '', parts[-1])
        activation = parts[-2]
        
        # Mimari kısmını birleştir (birden fazla kısım olabilir)
        architecture = '_'.join(parts[:-2])
        
        return architecture, activation, optimizer
    except Exception as e:
        print(f"Hata: {model_dir} modeli işlenirken bir sorun oluştu: {e}")
        return None, None, None

def get_best_metrics(results_path):
    """Results.csv dosyasından en iyi metrikleri alır."""
    try:
        if not os.path.exists(results_path):
            print(f"Uyarı: {results_path} dosyası bulunamadı")
            return None
            
        df = pd.read_csv(results_path)
        
        # Boş veya bozuk CSV dosyası kontrolü
        if df.empty:
            print(f"Uyarı: {results_path} dosyası boş")
            return None
            
        # En iyi mAP@0.5 değerini bul
        map_columns = [col for col in df.columns if 'mAP_0.5' in col]
        
        if not map_columns:
            print(f"Uyarı: {results_path} dosyasında mAP sütunu bulunamadı")
            return None
            
        map_column = map_columns[0]  # İlk bulunan mAP sütununu kullan
        
        # Sütun var mı kontrol et
        if map_column in df.columns:
            # NaN değerleri kontrol et
            if df[map_column].isna().all():
                print(f"Uyarı: {results_path} dosyasındaki {map_column} sütunu tüm değerler NaN")
                return None
                
            best_map_idx = df[map_column].idxmax()
            best_map50 = df.loc[best_map_idx, map_column]
            
            # epoch sütunu var mı kontrol et
            if 'epoch' in df.columns:
                best_epoch = df.loc[best_map_idx, 'epoch']
            else:
                # Yoksa sıra numarası kullan
                best_epoch = best_map_idx
            
            # Aynı epoch'taki diğer metrikleri al - sütun isimlerinde küçük farklılıklar olabilir
            map50_95_columns = [col for col in df.columns if '0.5:0.95' in col]
            precision_columns = [col for col in df.columns if 'precision' in col.lower()]
            recall_columns = [col for col in df.columns if 'recall' in col.lower()]
            
            best_map50_95 = df.loc[best_map_idx, map50_95_columns[0]] if map50_95_columns else 0
            best_precision = df.loc[best_map_idx, precision_columns[0]] if precision_columns else 0
            best_recall = df.loc[best_map_idx, recall_columns[0]] if recall_columns else 0
            
            # Son epoch'taki kayıp değerlerini al
            last_idx = df.index[-1]
            
            loss_columns = {
                'box_loss': [col for col in df.columns if 'box_loss' in col],
                'cls_loss': [col for col in df.columns if 'cls_loss' in col],
                'dfl_loss': [col for col in df.columns if 'dfl_loss' in col]
            }
            
            last_box_loss = df.loc[last_idx, loss_columns['box_loss'][0]] if loss_columns['box_loss'] else 0
            last_cls_loss = df.loc[last_idx, loss_columns['cls_loss'][0]] if loss_columns['cls_loss'] else 0
            last_dfl_loss = df.loc[last_idx, loss_columns['dfl_loss'][0]] if loss_columns['dfl_loss'] else 0
            
            return {
                'best_map50': best_map50,
                'best_map50_95': best_map50_95,
                'best_precision': best_precision,
                'best_recall': best_recall,
                'best_epoch': best_epoch,
                'last_box_loss': last_box_loss,
                'last_cls_loss': last_cls_loss,
                'last_dfl_loss': last_dfl_loss,
                'epochs': len(df)
            }
        else:
            print(f"Uyarı: {results_path} dosyasında mAP sütunu bulunamadı")
            return None
    except Exception as e:
        print(f"Hata: {results_path} dosyası okunurken bir sorun oluştu: {e}")
        return None

def extract_model_size(weights_path):
    """Ağırlık dosyasının boyutunu MB cinsinden döndürür."""
    try:
        if os.path.exists(weights_path):
            size_bytes = os.path.getsize(weights_path)
            size_mb = size_bytes / (1024 * 1024)
            return size_mb
        else:
            return None
    except Exception as e:
        print(f"Hata: {weights_path} dosyası okunurken bir sorun oluştu: {e}")
        return None

def create_visualizations(df):
    """Modellerin performans karşılaştırmalarını gösteren grafikler oluşturur."""
    if df.empty:
        print("Görselleştirme için yeterli veri yok!")
        return
        
    # Görselleştirme dizinini oluştur
    viz_dir = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Tema ve stil ayarları
    plt.style.use('ggplot')
    sns.set_theme(style="whitegrid")
    
    # 1. Mimarilere göre mAP@0.5 karşılaştırması
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x='architecture', y='best_map50', hue='activation', data=df)
    chart.set_title('Mimarilere ve Aktivasyon Fonksiyonlarına Göre mAP@0.5 Performansı', fontsize=16)
    chart.set_xlabel('Model Mimarisi', fontsize=14)
    chart.set_ylabel('mAP@0.5', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'map50_by_architecture.png'), dpi=300)
    plt.close()
    
    # 2. Optimizasyon algoritmalarına göre mAP@0.5 karşılaştırması
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x='optimizer', y='best_map50', hue='activation', data=df)
    chart.set_title('Optimizasyon Algoritmalarına Göre mAP@0.5 Performansı', fontsize=16)
    chart.set_xlabel('Optimizasyon Algoritması', fontsize=14)
    chart.set_ylabel('mAP@0.5', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'map50_by_optimizer.png'), dpi=300)
    plt.close()
    
    # 3. Aktivasyon fonksiyonlarına göre karşılaştırma
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x='activation', y='best_map50', hue='architecture', data=df)
    chart.set_title('Aktivasyon Fonksiyonlarına Göre mAP@0.5 Performansı', fontsize=16)
    chart.set_xlabel('Aktivasyon Fonksiyonu', fontsize=14)
    chart.set_ylabel('mAP@0.5', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'map50_by_activation.png'), dpi=300)
    plt.close()
    
    # 4. Precision-Recall ilişkisi
    plt.figure(figsize=(10, 8))
    chart = sns.scatterplot(x='best_recall', y='best_precision', hue='architecture', 
                           size='best_map50', sizes=(50, 200), data=df)
    chart.set_title('Kesinlik-Hassasiyet İlişkisi', fontsize=16)
    chart.set_xlabel('Hassasiyet (Recall)', fontsize=14)
    chart.set_ylabel('Kesinlik (Precision)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'precision_recall.png'), dpi=300)
    plt.close()
    
    # 5. Model sayılarının dağılımı
    plt.figure(figsize=(12, 6))
    counts = df.groupby(['architecture', 'activation']).size().unstack(fill_value=0)
    counts.plot(kind='bar', stacked=True)
    plt.title('Mimari ve Aktivasyon Fonksiyonlarına Göre Model Sayıları', fontsize=16)
    plt.xlabel('Model Mimarisi', fontsize=14)
    plt.ylabel('Model Sayısı', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'model_counts.png'), dpi=300)
    plt.close()
    
    print(f"Görselleştirmeler {viz_dir} klasörüne kaydedildi.")

def main():
    print(f"Modeller {TRAIN_DIR} dizininden okunuyor...")
    
    # TRAIN_DIR var mı kontrol et
    if not os.path.exists(TRAIN_DIR):
        print(f"Hata: {TRAIN_DIR} dizini bulunamadı!")
        return
        
    # TRAIN_DIR içerisini listele ve varlığını doğrula
    try:
        dir_contents = os.listdir(TRAIN_DIR)
        print(f"Bulunan dosya/klasör sayısı: {len(dir_contents)}")
        print(f"İçerik örnekleri: {', '.join(dir_contents[:5]) if dir_contents else 'Boş dizin'}")
    except Exception as e:
        print(f"Hata: {TRAIN_DIR} dizini listelenirken bir sorun oluştu: {e}")
        return
    
    all_results = []
    
    # runs/train altındaki her bir model klasörünü dolaş
    for model_dir in sorted(os.listdir(TRAIN_DIR)):
        dir_path = os.path.join(TRAIN_DIR, model_dir)
        
        # Klasör değilse veya history klasörüyse atla
        if not os.path.isdir(dir_path) or model_dir == "history":
            continue
        
        print(f"İşleniyor: {model_dir}")
        
        # Model bilgilerini çıkar
        architecture, activation, optimizer = extract_model_info(model_dir)
        if not all([architecture, activation, optimizer]):
            print(f"Uyarı: {model_dir} model bilgileri çıkarılamadı, atlanıyor...")
            continue
        
        # Results.csv dosyasını kontrol et
        results_path = os.path.join(dir_path, "results.csv")
        if not os.path.exists(results_path):
            print(f"Uyarı: {model_dir} için results.csv dosyası bulunamadı, atlanıyor...")
            continue
            
        # Metrikleri al
        metrics = get_best_metrics(results_path)
        if not metrics:
            print(f"Uyarı: {model_dir} için metrikler çıkarılamadı, atlanıyor...")
            continue
            
        # Model boyutunu al
        weights_path = os.path.join(dir_path, "weights", "best.pt")
        model_size = extract_model_size(weights_path)
        
        # Sonuçları kaydet
        all_results.append({
            'model_name': model_dir,
            'architecture': architecture,
            'activation': activation,
            'optimizer': optimizer,
            'model_size_mb': model_size if model_size else 0,
            **metrics
        })
    
    # Sonuçları DataFrame'e dönüştür
    if all_results:
        print(f"Toplam {len(all_results)} model başarıyla işlendi.")
        df = pd.DataFrame(all_results)
        
        # Daha iyi bir sıralama ve görünüm için sütunları düzenle
        columns_order = [
            'model_name', 'architecture', 'activation', 'optimizer', 'model_size_mb',
            'best_map50', 'best_map50_95', 'best_precision', 'best_recall', 
            'best_epoch', 'epochs', 'last_box_loss', 'last_cls_loss', 'last_dfl_loss'
        ]
        df = df[columns_order]
        
        # Mimari, aktivasyon fonksiyonu ve optimizasyona göre sırala
        df = df.sort_values(['architecture', 'activation', 'optimizer'])
        
        # Sayısal sütunları yuvarlama
        for col in ['best_map50', 'best_map50_95', 'best_precision', 'best_recall', 
                   'last_box_loss', 'last_cls_loss', 'last_dfl_loss', 'model_size_mb']:
            df[col] = df[col].round(4)
            
        # Sonuçları CSV olarak kaydet
        csv_path = os.path.join(OUTPUT_DIR, "model_comparison_table.csv")
        df.to_csv(csv_path, index=False)
        
        # Markdown tablosu oluştur
        md_table = "# YOLOv9 Eğitilmiş Model Karşılaştırma Tablosu\n\n"
        md_table += "| Model Mimarisi | Aktivasyon Fonksiyonu | Optimizasyon | Model Boyutu (MB) | mAP@0.5 | mAP@0.5:0.95 | Kesinlik | Hassasiyet | En İyi Epok | Toplam Epok | Box Loss | Cls Loss | DFL Loss |\n"
        md_table += "|---------------|------------------------|--------------|-------------------|---------|--------------|----------|------------|------------|-------------|----------|----------|----------|\n"
        
        # Satırları ekle
        for _, row in df.iterrows():
            md_table += f"| {row['architecture']} | {row['activation']} | {row['optimizer']} | {row['model_size_mb']:.2f} | {row['best_map50']:.4f} | {row['best_map50_95']:.4f} | {row['best_precision']:.4f} | {row['best_recall']:.4f} | {int(row['best_epoch'])} | {int(row['epochs'])} | {row['last_box_loss']:.4f} | {row['last_cls_loss']:.4f} | {row['last_dfl_loss']:.4f} |\n"
        
        # Markdown dosyasını kaydet
        md_path = os.path.join(OUTPUT_DIR, "model_comparison_table.md")
        with open(md_path, "w") as f:
            f.write(md_table)
            
        # LaTeX tablosu oluştur (tez için)
        latex_table = "\\begin{table}[htbp]\n"
        latex_table += "\\centering\n"
        latex_table += "\\caption{YOLOv9 Eğitilmiş Model Karşılaştırma Tablosu}\n"
        latex_table += "\\label{tab:model-comparison}\n"
        latex_table += "\\resizebox{\\textwidth}{!}{\n"
        latex_table += "\\begin{tabular}{lllrrrrrrrrrr}\n"
        latex_table += "\\toprule\n"
        latex_table += "Model Mimarisi & Aktivasyon & Optimizasyon & Boyut (MB) & mAP@0.5 & mAP@0.5:0.95 & Kesinlik & Hassasiyet & En İyi Epok & Toplam Epok & Box Loss & Cls Loss & DFL Loss \\\\\n"
        latex_table += "\\midrule\n"
        
        # Satırları ekle
        for _, row in df.iterrows():
            latex_table += f"{row['architecture']} & {row['activation']} & {row['optimizer']} & {row['model_size_mb']:.2f} & {row['best_map50']:.4f} & {row['best_map50_95']:.4f} & {row['best_precision']:.4f} & {row['best_recall']:.4f} & {int(row['best_epoch'])} & {int(row['epochs'])} & {row['last_box_loss']:.4f} & {row['last_cls_loss']:.4f} & {row['last_dfl_loss']:.4f} \\\\\n"
        
        latex_table += "\\bottomrule\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "}\n"
        latex_table += "\\end{table}\n"
        
        # LaTeX dosyasını kaydet
        latex_path = os.path.join(OUTPUT_DIR, "model_comparison_table.tex")
        with open(latex_path, "w") as f:
            f.write(latex_table)
            
        # Görselleştirmeleri oluştur
        create_visualizations(df)
            
        print("Karşılaştırma tablosu başarıyla oluşturuldu!")
        print(f"CSV: {csv_path}")
        print(f"Markdown: {md_path}")
        print(f"LaTeX: {latex_path}")
    else:
        print("İşlenecek sonuç bulunamadı!")
        print(f"Dizin: {TRAIN_DIR}")
        print("Lütfen dizin yolunu kontrol edin veya modellerin doğru formatta olduğundan emin olun.")

if __name__ == "__main__":
    main()