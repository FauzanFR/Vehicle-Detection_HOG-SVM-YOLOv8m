# Vehicle Detection — HOG+SVM & YOLOv8m

Proyek ini berisi dua pendekatan berbeda untuk deteksi kendaraan pada dataset yang sama (5 kelas: Bus, Car, Motorcycle, Pickup, Truck), menggunakan arsitektur klasik (HOG+SVM) dan modern (YOLOv8m).

---

## Struktur File

```
├── vehicle_detection_yolov8.ipynb    # Notebook 1 — YOLOv8m (Kaggle)
├── vehicle_detection_hog_svm.ipynb   # Notebook 2 — HOG + SVM (lokal)
└── README.md
```

---

## Notebook 1 — YOLOv8m (`vehicle_detection_yolov8.ipynb`)

**Platform:** Kaggle (GPU)  
**Dataset:** [pkdarabi/vehicle-detection-image-dataset](https://www.kaggle.com/datasets/pkdarabi/vehicle-detection-image-dataset) — versi color (No_Apply_Grayscale)

### Pipeline

| Step | Deskripsi |
|------|-----------|
| 1. Install & Import | Ultralytics, Albumentations, seaborn, scikit-image |
| 2. Configuration | Paths Kaggle, hyperparameter, class config |
| 3. Utility Functions | YAML fixer, plot saver |
| 4. Custom Augmentation | RandomResizedCrop + MedianBlur + BrightnessContrast + HFlip (via Albumentations) |
| 5. Training Config | Class-balanced loss weighting (inverse frequency), YOLOv8 aug params |
| 6. Training | `model.train()` dengan callback custom loss + multiscale |
| 7. Evaluation | Validation + test split, mAP50 / mAP50-95 / Precision / Recall |
| 8. Post-Processing | Soft-NMS + per-class confidence threshold + geometry filter → JSON export |
| 9. Results Summary | Tabel metrik + bar chart |
| 10. Visualisation | 6 sample prediksi test set |
| 11. Export | ZIP semua output untuk download dari Kaggle |

### Teknik Utama

- **Class-Balanced Weighting:** Dataset sangat imbalanced (Car 73%, Bus 1%). Bobot per kelas dihitung dari inverse frequency dan diterapkan via callback `on_train_batch_end`.
- **Soft-NMS:** Menggantikan hard NMS untuk mengurangi suppression berlebihan pada kendaraan yang berdekatan.
- **Per-class confidence threshold:** Kelas minoritas (Bus, Truck) menggunakan threshold lebih rendah (0.15) vs kelas mayoritas (Car = 0.30).

### Hyperparameter

```python
EPOCHS     = 50
BATCH_SIZE = 16
IMG_SIZE   = 640
PATIENCE   = 15
SEED       = 42
MULTISCALE = True
```

### Requirements

```
ultralytics
albumentations
seaborn
scikit-image
torch  # tersedia di Kaggle GPU env
```

---

## Notebook 2 — HOG + SVM (`vehicle_detection_hog_svm.ipynb`)

**Platform:** Lokal  
**Dataset:** YOLOv9 format (grayscale), path `dataset/Apply_Grayscale/...`

### Pipeline

| Step | Deskripsi |
|------|-----------|
| 1. Imports & Config | OpenCV, sklearn, imbalanced-learn, path config |
| 2. Load Labels | Parser YOLO-format label ke DataFrame |
| 3. EDA: Class Dist. | Distribusi kelas per split + total |
| 4. EDA: Box Count | Histogram jumlah box per image |
| 5. EDA: Box Size | Scatter w×h, distribusi area, aspect ratio |
| 6. Outlier Filtering | Filter berdasarkan area (p1) dan aspect ratio (p1–p99) |
| 7. Annotation Viz | 3×3 grid sampel anotasi ground truth |
| 8. Sample Extraction | Crop positif dari GT box + augmentasi; crop negatif random (IoU < 0.2) |
| 9. HOG + PCA | HOG 64×64 (1764-dim) → PCA 200 komponen |
| 10. SVM Training | SMOTE balancing + GridSearchCV (RBF, C/gamma) + StratifiedKFold |
| 11. Evaluation | Classification report, confusion matrix, PR curve, optimal threshold |
| 12. Hard Negative Mining | 3 iterasi bootstrap: FP dari sliding window → retrain SVM |
| 13. Test Evaluation | Sliding window detection vs GT boxes (TP/FP/FN → Precision/Recall/F1) |
| 14. Detection Viz | 6 sampel validasi: hijau = GT, merah = prediksi |
| 15. Video Detection | Frame-by-frame detection dengan skip_frames |

### Teknik Utama

- **CLAHE preprocessing:** Meningkatkan kontras lokal sebelum HOG extraction, terutama berguna untuk gambar grayscale.
- **Hard Negative Mining:** 3 iterasi bootstrapping — false positive dari sliding window dimasukkan sebagai sampel negatif baru untuk retraining SVM.
- **Decision function threshold:** Threshold optimal dicari dengan grid search F1-maximisation (bukan default 0.5 probability cutoff).
- **Geometry filter:** Setelah NMS, deteksi difilter berdasarkan area (1200–150000 px²), aspect ratio (0.5–2.5), edge density (0.03–0.7), dan variance (>150) untuk membuang deteksi trivial.

### HOG Config

```python
WINDOW_SIZE  = (64, 64)
BLOCK_SIZE   = (16, 16)
BLOCK_STRIDE = (8, 8)
CELL_SIZE    = (8, 8)
N_BINS       = 9
# → 1764 fitur per sample → PCA 200 komponen
```

### Requirements

```
opencv-python
numpy
pandas
matplotlib
scikit-learn
imbalanced-learn
```

Install:
```bash
pip install opencv-python numpy pandas matplotlib scikit-learn imbalanced-learn
```

---

## Perbandingan Pendekatan

| Aspek | HOG + SVM | YOLOv8m |
|-------|-----------|---------|
| Jenis model | Classical ML | Deep learning (CNN) |
| Input | Grayscale 64×64 crop | Color 640×640 |
| Detection type | Sliding window (binary) | End-to-end detection |
| Class output | Vehicle / Background | 5 kelas spesifik |
| Training time | Menit (CPU) | ~50 epoch GPU |
| Inference speed | Lambat (sliding window) | Real-time |
| Kelebihan | Interpretable, ringan | Akurat, multi-class |

---

## Dataset

Kedua notebook menggunakan dataset kendaraan yang sama, hanya berbeda preprocessing:

- **YOLOv8 notebook:** versi color (RGB), format YOLOv8
- **HOG+SVM notebook:** versi grayscale, format YOLOv9

**Distribusi kelas (approx):**  
Car ≈ 73% · Motorcycle ≈ 14% · Pickup ≈ 10% · Truck ≈ 2% · Bus ≈ 1%

Dataset sangat imbalanced — kedua notebook menangani ini secara eksplisit (class weights pada YOLO, SMOTE + class_weight='balanced' pada SVM).
