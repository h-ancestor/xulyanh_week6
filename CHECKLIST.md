# Đối chiếu sản phẩm với đề bài

## Bài 1 — Đặc trưng thống kê và màu sắc

| Yêu cầu | Tệp minh chứng |
|---|---|
| 3 lớp táo/chuối/cam, tối thiểu 50 ảnh/lớp | `dataset/` |
| Resize 128×128; đổi RGB → HSV; minh họa RGB, H/S/V, grayscale | `week6_solution.py`, `results/fruit_preprocessing.png` |
| Mean + Standard Deviation từng kênh RGB và HSV | `results/fruit_feature_table.csv` |
| Histogram HSV (8 bins/kênh) và entropy ảnh xám | `results/fruit_feature_table.csv` |
| Vector đặc trưng cho từng ảnh kèm nhãn | `results/fruit_feature_table.csv` (150 hàng) |
| Train/Test 80/20 phân tầng; chuẩn hóa fit trên Train | `week6_solution.py` dùng `train_test_split` và pipeline `StandardScaler` |
| TN1 RGB stats/KNN; TN2 HSV stats/KNN; TN3 histogram HSV/SVM; TN4 tất cả/SVM | `results/fruit_metrics.csv`, `results/fruit_confusion_matrices.png` |
| Accuracy, Precision, Recall, F1 và confusion matrix | `results/fruit_metrics.csv`, `results/fruit_results.json`, hình ma trận |
| Dự đoán ảnh test chưa dùng trong Train, có nhãn thật/dự đoán | `results/fruit_test_predictions.csv`, `results/fruit_test_predictions.png` |

## Bài 2 — Hình học và HOG

| Yêu cầu | Tệp minh chứng |
|---|---|
| 3 lớp tròn/vuông/tam giác, tối thiểu 50 ảnh/lớp | `dataset_shapes/` (60 ảnh/lớp) |
| Grayscale, Gaussian blur, Otsu, Canny, Contour | `week6_solution.py` |
| Minh họa ảnh gốc, grayscale, Otsu, Canny, contour, HOG | `results/shape_preprocessing.png` |
| Diện tích, chu vi, kích thước, W/H, độ tròn, mật độ biên | `results/shape_feature_table.csv` |
| HOG vector và nhãn cho từng ảnh | `results/shape_feature_table.csv` (180 hàng, 1.764 chiều HOG) |
| KNN, SVM, Random Forest trên hình học/HOG/kết hợp | `results/shape_metrics.csv` |
| Accuracy, Precision, Recall, F1 và confusion matrix | `results/shape_metrics.csv`, `results/shape_results.json`, hình ma trận |
| Dự đoán ảnh độc lập và phân tích trường hợp phân loại sai | `results/shape_unseen_predictions.csv`; phần 4.1–4.2 trong báo cáo |

## Tài liệu nộp

- Source code: `week6_solution.py`, `download_dataset.py`
- Notebook: `week6_ml.ipynb`
- Cài đặt và tái tạo: `requirements.txt`, `README.md`

Đề PDF và báo cáo Word được giữ bên ngoài thư mục GitHub theo yêu cầu. Có thể tái tạo báo cáo Word bằng cách chạy `python week6_solution.py`; file `.docx` được `.gitignore` loại khỏi Git.
