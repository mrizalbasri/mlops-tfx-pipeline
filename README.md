# Proyek Pengembangan dan Pengoperasian Sistem Machine Learning
**Dicoding Submission - Machine Learning Operations (MLOps)**

* **Nama:** M. Rizal Basri
* **Email:** m.basri@student.president.ac.id
* **Username Dicoding:** rizalbasri
* **Pipeline Name:** `rizalbasri-pipeline`
* **Dataset:** Heart Disease UCI Dataset (Klasifikasi Biner)
* **Status Penilaian Pylint:** 10.00 / 10.00

---

## 1. Informasi Terkait Dataset
Dataset yang digunakan dalam proyek ini adalah **Heart Disease Dataset** yang bersumber dari UCI Machine Learning Repository (Cleveland Database). Dataset ini berisi catatan medis klinis pasien untuk memprediksi apakah seorang pasien memiliki indikasi penyakit jantung atau tidak (klasifikasi biner).

* **Total Sampel:** 303 baris data klinis.
* **Jumlah Fitur:** 13 fitur masukan (fitur numerik & kategorikal) dan 1 label target.
* **Karakteristik Missing Value:** 0 missing value (data bersih).
* **Fitur Input:**
  1. `age`: Usia pasien dalam tahun (numerik).
  2. `sex`: Jenis kelamin (1 = laki-laki, 0 = perempuan).
  3. `cp`: Tipe nyeri dada / *Chest Pain* (nilai 0 - 3).
  4. `trestbps`: Tekanan darah saat istirahat dalam mm Hg (numerik).
  5. `chol`: Kolesterol serum dalam mg/dl (numerik).
  6. `fbs`: Gula darah puasa > 120 mg/dl (1 = ya, 0 = tidak).
  7. `restecg`: Hasil elektrokardiografi istirahat (nilai 0 - 2).
  8. `thalach`: Detak jantung maksimum yang tercapai (numerik).
  9. `exang`: Angina yang diinduksi oleh aktivitas fisik (1 = ya, 0 = tidak).
  10. `oldpeak`: Depresi ST yang diinduksi oleh aktivitas fisik relatif terhadap istirahat (float).
  11. `slope`: Kemiringan segmen ST puncak aktivitas fisik (nilai 0 - 2).
  12. `ca`: Jumlah pembuluh darah utama yang diwarnai dengan fluoroskopi (nilai 0 - 3).
  13. `thal`: Thalasemia (0 = normal, 1 = fixed defect, 2 = reversible defect).
* **Target Label:**
  * `target`: Indikasi penyakit jantung (`0` = Healthy / Normal, `1` = Heart Disease Detected).

---

## 2. Persoalan yang Ingin Diselesaikan
Penyakit kardiovaskular (jantung) merupakan salah satu penyebab kematian tertinggi di dunia. Diagnosis dini yang akurat dan tepat waktu sangat esensial untuk menyelamatkan nyawa pasien dan menentukan intervensi medis yang tepat. 

Secara konvensional, model machine learning sering kali hanya berhenti pada tahap eksperimentasi di notebook (model statis), sehingga menghadapi kendala operasional yang serius ketika dibawa ke produksi (*training-serving skew*, ketiadaan validasi skema otomatis, tidak adanya pipeline retraining terotomatisasi, serta minimnya observabilitas performa model saat menerima data pasien baru di lingkungan produksi).

Oleh karena itu, proyek ini menyelesaikan persoalan tersebut dengan membangun **Sistem Machine Learning Pipeline End-to-End** berbasis **MLOps** yang andal, dapat diproduksi secara konsisten, di-deploy ke lingkungan komputasi cloud, serta dipantau kesehatannya secara *real-time*.

---

## 3. Solusi Machine Learning & Target yang Ingin Dicapai
Solusi yang dibangun mencakup arsitektur MLOps terintegrasi:
1. **Machine Learning Pipeline Terstandarisasi:** Menggunakan **TensorFlow Extended (TFX)** yang diorkestrasi secara modular menggunakan **Apache Beam** (`BeamDagRunner`).
2. **Komponen Pipeline Lengkap (9 Komponen):** Meliputi `CsvExampleGen`, `StatisticsGen`, `SchemaGen`, `ExampleValidator`, `Transform`, `Trainer`, `Resolver` (Latest Blessed Model), `Evaluator` (TFMA), dan `Pusher`.
3. **Penyimpanan Artefak & Metadata:** Menyimpan seluruh riwayat eksekusi, garis keturunan data (*data lineage*), dan artefak ke dalam **ML Metadata (MLMD)**.
4. **Target Performa:**
   * Akurasi pelatihan > 90% dan AUC validasi > 0.85.
   * Model wajib lolos evaluasi otomatis TFMA dengan status **BLESSED** sebelum diizinkan untuk di-push ke folder serving.
   * Latensi inferensi API di cloud ditargetkan < 50 ms untuk setiap request pasien.

---

## 4. Metode Pengolahan Data, Arsitektur Model, & Metrik Evaluasi

### A. Pengolahan Data (Transform Component - `modules/transform.py`)
Preprocessing dilakukan menggunakan pustaka **TensorFlow Transform (`tft`)**, sehingga logika preprocessing menjadi bagian integral dari graph model Keras yang diekspor (*preventing training-serving skew*):
* **Fitur Numerik** (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`): Dinormalisasi ke skala standar menggunakan `tft.scale_to_z_score`.
* **Fitur Kategorikal** (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`): Dikonversi ke tipe `tf.float32` agar siap dikombinasikan ke dalam dense layer.
* **Target Label** (`target`): Dikonversi ke tipe `tf.int64`.

### B. Arsitektur Model (Trainer Component - `modules/trainer.py`)
Model yang dibangun adalah **Deep Neural Network (DNN)** berbasis Keras:
* **Input Layer:** Multi-input tensor untuk setiap fitur yang telah ditransformasi.
* **Concatenation Layer:** Menggabungkan seluruh representasi fitur numerik dan kategorikal.
* **Hidden Layer 1:** Dense layer 64 unit dengan aktivasi `ReLU`.
* **Regularization:** Dropout layer dengan *rate* 0.2 untuk mencegah *overfitting*.
* **Hidden Layer 2:** Dense layer 32 unit dengan aktivasi `ReLU`.
* **Regularization:** Dropout layer dengan *rate* 0.1.
* **Output Layer:** Dense layer 1 unit dengan fungsi aktivasi `Sigmoid` (menghasilkan probabilitas antara 0.0 hingga 1.0).
* **Optimizer & Loss:** Adam optimizer (learning rate = 0.001) dengan `BinaryCrossentropy` loss function.
* **Serving Signature:** Menyertakan signature `serving_default` yang menerima string serialized `tf.train.Example` dan mengeksekusi layer transformasi TFT secara otomatis.

### C. Metrik Evaluasi (Evaluator Component)
Evaluasi performa model dilakukan menggunakan **TensorFlow Model Analysis (TFMA)**:
* `BinaryAccuracy`: Menilai akurasi klasifikasi biner dengan ambang batas minimal `0.5`.
* `AUC (Area Under ROC Curve)`: Mengukur kemampuan diskriminasi model dalam membedakan pasien sehat dan berisiko penyakit jantung.
* `ChangeThreshold`: Memastikan model baru memiliki performa yang setara atau lebih baik daripada baseline model sebelumnya (`direction: HIGHER_IS_BETTER`).

---

## 5. Informasi Terkait Performa Model Machine Learning
Berdasarkan hasil pelatihan dan evaluasi melalui TFX Pipeline dengan Apache Beam:
* **Training Accuracy:** **95.56%**
* **Training AUC:** **0.9885**
* **Validation Accuracy:** **80.47%**
* **Validation AUC:** **0.8641**
* **Evaluator Status:** **BLESSED**  
  Model berhasil melampaui seluruh ambang batas validasi TFMA, sehingga disetujui untuk dipush oleh komponen `Pusher` ke direktori `serving_model_dir/`.

---

## 6. Opsi Model Deployment & Platform yang Digunakan
Sistem model serving dikemas ke dalam kontainer **Docker** independen dan siap dijalankan di berbagai platform komputasi cloud:
* **Framework Serving:** **FastAPI** + **Uvicorn** (asynchronous, performa tinggi, validasi skema otomatis, dan integrasi Swagger UI).
* **Containerization:** Berbasis image `python:3.10-slim` dengan konfigurasi `Dockerfile` produksi.
* **Cloud Platform:** **Railway** / **Render** / **Cloud Container Registry**.
* **Model Loading:** Menggunakan `tf.saved_model.load()` yang secara dinamis memuat artefak `SavedModel` versi terbaru dari hasil ekspor TFX Pusher.
* **Endpoint yang Disediakan:**
  * `GET /`: Informasi umum layanan API dan status model.
  * `GET /health`: Health-check endpoint untuk memverifikasi kesiapan model.
  * `POST /predict`: Menerima data klinis pasien, memvalidasi tipe data, melakukan serialisasi `tf.train.Example`, dan mengembalikan diagnosis inferensi beserta probabilitasnya.
  * `GET /metrics`: Endpoint metrik Prometheus yang memaparkan data latensi, jumlah request, dan status model.
  * `GET /docs`: Dokumentasi interaktif Swagger UI.

---

## 7. Tautan Web App Model Serving
* **URL Cloud Serving:** `https://rizalbasri-pipeline-production.up.railway.app`
* **URL Lokal:** `http://localhost:8000`
* **Dokumentasi API Swagger UI:** `http://localhost:8000/docs`
* **Endpoint Prediksi:** `http://localhost:8000/predict`
* **Endpoint Prometheus Metrics:** `http://localhost:8000/metrics`

---

## 8. Penjelasan Singkat Hasil Monitoring Sistem Machine Learning
Monitoring sistem diimplementasikan menggunakan **Prometheus** (dan visualisasi **Grafana**) untuk memastikan keandalan sistem produksi:
* **Metrik yang Dipantau:**
  1. `http_requests_total`: Menghitung total volume panggilan HTTP berdasarkan metode (`GET`/`POST`), path (`/predict`, `/health`, `/metrics`), dan kode status HTTP (`200`, `500`).
  2. `http_request_duration_seconds`: Mengukur distribusi latensi pemrosesan inferensi pasien (P95 berada di angka **8.42 ms**, jauh di bawah batas SLA 50 ms).
  3. `model_predictions_total`: Memantau distribusi hasil prediksi model secara real-time (`Heart Disease Detected` vs `Healthy / Normal`).
  4. `model_prediction_value`: Histogram persebaran skor probabilitas pasien yang masuk.
  5. `model_loaded_status`: Gauge liveness indikator keberadaan model di memori (1.0 = Sehat/Aktif).
* **Scrape Configuration (`monitoring/prometheus.yml`):** Dikonfigurasi dengan interval scrape setiap 5 detik untuk mendeteksi lonjakan trafik atau degradasi performa secara cepat.

---

## 9. Penilaian Kualitas Kode (Pylint 10.00 / 10.00)
Seluruh kode pada direktori `modules/` (`transform.py`, `trainer.py`, `components.py`) telah diuji dan diverifikasi menggunakan **Pylint** dengan skor sempurna:
```text
--------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)
--------------------------------------------------------------------
```
Bukti screenshot terlampir pada `rizalbasri-pylint.png`.

---

## 10. Struktur Berkas Proyek Submission

```text
rizalbasri-pipeline/
├── .gitignore                          # Aturan git ignore (mengabaikan venv, cache, secrets)
├── README.md                           # Dokumentasi komprehensif proyek (format-dokumentasi-2)
├── requirements.txt                    # Daftar dependencies Python yang digunakan
├── Dockerfile                          # Dockerfile produksi untuk deployment model serving
├── docker-compose.yml                  # Konfigurasi orkestrasi Serving API + Prometheus + Grafana
├── pipeline.py                         # Skrip eksekusi pipeline TFX dengan Apache Beam
├── app.py                              # REST API model serving dengan Prometheus metrics
├── sample_request.json                 # Contoh payload request inferensi pasien
│
├── data/                               # Direktori dataset input ExampleGen
│   └── heart.csv
│
├── modules/                            # Direktori modul pipeline (Saran 2 - Pylint 10/10)
│   ├── __init__.py
│   ├── transform.py                    # Preprocessing function TFT
│   ├── trainer.py                      # Keras DNN model & training run_fn
│   └── components.py                   # Inisialisasi 9 komponen TFX
│
├── serving_model_dir/                  # Direktori artefak model serving yang diekspor Pusher
│   └── 1789577747/
│       ├── saved_model.pb
│       ├── keras_metadata.pb
│       ├── variables/
│       └── assets/
│
├── monitoring/                         # Direktori konfigurasi Prometheus (Kriteria 4)
│   ├── Dockerfile                      # Dockerfile untuk Prometheus server
│   ├── prometheus.yml                  # Konfigurasi target scraping metrics
│   └── prometheus.config               # Salinan konfigurasi prometheus
│
├── rizalbasri-pipeline.ipynb           # Notebook utama eksekusi TFX Pipeline (sudah dijalankan)
├── notebook.ipynb                      # Salinan rizalbasri-pipeline.ipynb untuk kemudahan review
├── rizalbasri-testing.ipynb            # Notebook pengujian request inferensi (Saran 3)
│
├── rizalbasri-pylint.png               # Screenshot penilaian Pylint 10/10 modul (Saran 2)
├── rizalbasri-deployment.png           # Screenshot bukti deployment sistem di cloud (Kriteria 3)
├── rizalbasri-monitoring.png           # Screenshot dashboard Prometheus (Kriteria 4)
├── rizalbasri-grafana-dashboard.png    # Screenshot dashboard Grafana (Saran 4)
│
└── screenshots/                        # Salinan arsip seluruh screenshot submission
    ├── rizalbasri-pylint.png
    ├── rizalbasri-deployment.png
    ├── rizalbasri-monitoring.png
    └── rizalbasri-grafana-dashboard.png
```

---

## 11. Panduan Menjalankan Proyek

### A. Menjalankan TFX Pipeline
Pastikan virtual environment telah aktif:
```powershell
.venv\Scripts\activate
python pipeline.py
```
Atau buka dan jalankan seluruh cell pada `rizalbasri-pipeline.ipynb`.

### B. Menjalankan Model Serving API
```powershell
python app.py
```
Akses Swagger UI pada `http://localhost:8000/docs` untuk menguji endpoint secara interaktif.

### C. Menjalankan Monitoring (Prometheus & Grafana)
Menggunakan Docker Compose:
```bash
docker-compose up --build
```
* **Serving API:** `http://localhost:8000`
* **Prometheus UI:** `http://localhost:9090`
* **Grafana UI:** `http://localhost:3000`

### D. Menjalankan Pengujian Inferensi
Buka dan jalankan seluruh cell pada notebook `rizalbasri-testing.ipynb`.
