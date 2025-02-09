# Deskripsi Umum
Repository ini berisi kumpulan proyek selama mengikuti kursus Data Science di Udemy. 
Proyek-proyek ini mencakup berbagai topik penting dalam data science, mulai dari data preprocessing, exploratory data analysis (EDA), machine learning models, 
hingga visualisasi data menggunakan Python dan pustaka seperti Pandas, Matplotlib, Seaborn, dan Scikit-learn.

# teknologi yang digunakan
Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
Jupyter Notebook

## Prediksi harapan hidup menggunakan Algoritma XGBOOST (REGRESI)
### pengertian XGBOOST 
merupakan algoritma populer _surpervised learning_ yang mengimplementasikan algoritma _gradient boosted tress_ 
algoritma ini bekerja dengan membangun model untuk memperbaiki kesalahan yang dilakukan oleh model sebelumnya. 
### Permasalahan
Bagaimana kita dapat memprediksi harapan hidup berdasarkan berbagai faktor sosial, ekonomi, 
dan kesehatan seperti tingkat kematian bayi, konsumsi alkohol, tingkat pendidikan, 
dan komposisi pendapatan?
### Tujuan 
Membangun model prediksi yang dapat memperkirakan nilai harapan hidup (life expectancy) 
secara akurat dengan menggunakan XGBoost Regression Model. 
### Evaluation Metric
- R² (R-squared) – Mengukur seberapa baik model menjelaskan variabilitas data target. (97%)
- Mean Absolute Error (MAE) – Mengukur rata-rata kesalahan absolut antara prediksi dan data sebenarnya. (1.057)
- Root Mean Squared Error (RMSE) - Mengukur rata-rata kesalahan prediksi dalam satuan yang sama dengan target (2.62)
- Mean Squared Error (MSE) - rata-rata kuadrat dari selisih antara prediksi dan nilai sebenarnya. (2.621)
### Solusi Permasalahan
  1. Data Collection & Import
      - Mengimpor dataset Life_Expectancy_Data.csv menggunakan Pandas.
  2. Data Cleaning & Feature Engineering
     - Memeriksa missing values dan mengisinya menggunakan strategi tertentu.
     - Menghitung nilai minimum, rata-rata, dan maksimum dari fitur life expectancy.
     - Memeriksa penggunaan memori dari DataFrame.
  3. Data Visualization
     - Membuat histogram, pairplot, dan heatmap untuk memahami korelasi antar fitur.
     - Membuat scatterplot:
        a. Antara Income Composition of Resources dan Life Expectancy, dengan atribut hue berdasarkan Status.
        b. Antara Schooling dan Life Expectancy, dengan atribut hue berdasarkan Status.
     - Memberikan analisis dari plot yang dihasilkan.
   4. Model Training
      - Membagi data menjadi 80% data latih dan 20% data uji.
      - Melatih model regresi menggunakan XGBoost.
   5. Model Evaluation
      - Mengevaluasi performa model menggunakan metrik R² dan MAE.
      - Membuat plot antara hasil prediksi model dengan data sebenarnya.
###  Visualisasi  
#### Hasil prediksi model menunjukkan pencapaian positif meskipun ada beberapa outlier. namun model mampu memprediksi dengan nilai cukup tinggi
![image](https://github.com/user-attachments/assets/d746f25e-cf22-47fe-a227-08217d2e6ed1)

#### Prediksi _life expectancy_ dengan _Income Composition if Resource_ prediksi harapan hidup didominasi oleh negara berkembang "_developing_" 
semakin tinggi income maka harapan hidup nya lebih lama
![image](https://github.com/user-attachments/assets/6a894be4-735a-4558-915e-93928e1d171c)

### Kesimpulan 
1. Model XGBoost memberikan hasil prediksi yang cukup baik dalam memperkirakan nilai life expectancy, dengan skor R² yang tinggi.
2. Terdapat beberapa fitur yang memiliki pengaruh besar terhadap prediksi, seperti Schooling dan Income Composition of Resources.
3. Visualisasi scatterplot menunjukkan bahwa negara dengan status berkembang cenderung memiliki harapan hidup yang lebih rendah dibandingkan negara maju.
### Perbaikan selanjutnya 
- Meningkatkan performa model dengan melakukan hyperparameter tuning.
- Memasukkan lebih banyak fitur eksternal yang relevan, seperti akses layanan kesehatan atau indeks pembangunan manusia.
- Menggunakan metode regresi lain untuk perbandingan performa, seperti Random Forest atau Gradient Boosting.

## Prediksi Gagal Bayar Kartu Kredit menggunakan algoritma klasifikasi
### Permasalahan 
Bagaimana kita dapat memprediksi peluang gagal bayar kartu kredit berdasarkan data historis dan demografis pelanggan?
### Tujuan
Bank ingin mengidentifikasi klien yang berpotensi gagal bayar kartu kredit berdasarkan riwayat pembayaran dan data demografis mereka. Dengan memprediksi risiko gagal bayar, bank dapat mengurangi kerugian dan membuat strategi mitigasi risiko yang lebih efektif.
 dengan membangun model klasifikasi untuk memprediksi apakah pelanggan akan gagal membayar tagihan kartu kredit bulan berikutnya (default.payment.next.month).
 ### Matrik Evaluasi
 1. Accuracy – Mengukur seberapa sering prediksi model benar.
 2. Precision – Mengukur akurasi dari prediksi positif model.
 3. Recall (Sensitivity) – Mengukur seberapa baik model mendeteksi kasus gagal bayar.
 4. ROC Curve & AUC (Area Under Curve) – Menggambarkan performa model di semua threshold klasifikasi.
### Solusi
1. Data Preparation
   - Impor dataset UCI_Credit_Card.csv dan periksa isi datanya
   - Lakukan EDA (Exploratory Data Analysis), termasuk analisis statistik deskriptif dan visualisasi.
   - Periksa missing values dan lakukan penanganan jika ditemukan.
   - Data Splitting
2. Pisahkan data menjadi 80% data latih dan 20% data uji.
   - Model Training & Evaluation
   - Latih dan evaluasi beberapa model klasifikasi berikut:
      XGBoost
      Support Vector Machine (SVM)
      Naive Bayes
      Logistic Regression
      Random Forest
      K-Nearest Neighbors (KNN)
3. Model Comparison
   - Bandingkan performa model menggunakan ROC Curve dan AUC Score.
   - Tentukan model dengan performa terbaik berdasarkan hasil evaluasi.
### Visualisasi 
#### Plot kurva ROC untuk seluruh model dan hitung nilai AUC
- roc_curve: Menghitung False Positive Rate (FPR) dan True Positive Rate (TPR) dari probabilitas prediksi.
- auc: Menghitung Area Under Curve (AUC) untuk setiap model.
grafik dibawah menunjukkan bahwa penggunaan algoritma random forest untuk memprediksi kegagalan bayar kartu kredit menghasilkan prediksi lebih baik dibanding yang lain, karena memiliki bentuk yang hampir diagonal yang mengartikan bahwa model berhasil memprediksi dengan benar
![image](https://github.com/user-attachments/assets/da70d761-a27f-4649-a3a7-e56c765b5561)

### Kesimpulan 
- Model XGBoost menunjukkan performa terbaik dengan AUC Score yang paling tinggi, diikuti oleh Random Forest dan Logistic Regression.
- Variabel seperti riwayat pembayaran (PAY_0 – PAY_6) dan jumlah tagihan sebelumnya (BILL_AMT1 – BILL_AMT6) sangat mempengaruhi prediksi gagal bayar.
### Perbaikan selanjutnya
  1. Melakukan hyperparameter tuning untuk meningkatkan akurasi model.
  2. Mencoba pendekatan ensemble learning untuk menggabungkan hasil dari beberapa model.

