# Thực hành tuần 6 — Xử lý ảnh và Thị giác Robot

Đây là bộ mã nguồn, dữ liệu và kết quả để đẩy lên GitHub. Có thể mở `week6_ml.ipynb` để xem và chạy từng bước; `week6_solution.py` chạy lại pipeline và tái tạo các kết quả thí nghiệm. 

## Cấu trúc

```text
week6_submission/
├── dataset/                    # 50 ảnh/lớp: apple, banana, orange
├── dataset_shapes/             # 60 ảnh/lớp: circle, square, triangle
├── dataset_shapes_unseen/      # Ảnh hình dạng độc lập để minh họa dự đoán mới
├── results/                    # Bảng đặc trưng, metrics, dự đoán, biểu đồ
├── week6_ml.ipynb              # Notebook đã chạy, có kết quả hiển thị
├── week6_solution.py           # Source code pipeline
├── download_dataset.py         # Tải 50 ảnh mỗi loại quả
├── requirements.txt
├── NOTICE.md                   # Nguồn và giấy phép dữ liệu
└── CHECKLIST.md                # Đối chiếu từng yêu cầu đề
```



## Nội dung thí nghiệm

- **Bài 1:** 150 ảnh Fruits-360 (50 ảnh mỗi lớp); resize 128×128; đặc trưng RGB Mean/Std, HSV Mean/Std, histogram HSV 24 chiều và entropy ảnh xám. TN1–TN4 dùng KNN/SVM như bảng trong đề.
- **Bài 2:** 180 ảnh hình dạng tổng hợp (60 ảnh mỗi lớp); Grayscale, Gaussian blur, Otsu, Canny, Contour; 9 đặc trưng hình học và HOG 1.764 chiều. So sánh KNN, SVM và Random Forest trên hình học, HOG, kết hợp.
- Cả hai bài dùng chia Train/Test phân tầng 80/20 với seed 42; các scaler được fit trên Train trong pipeline.
- Tập hình dạng tổng hợp được tạo bằng mã nguồn để kiểm soát vị trí, kích thước, góc xoay và nhiễu. Điểm số Bài 2 phản ánh tập tổng hợp này.

## Kết quả đã chạy

Các cấu hình Bài 1 đạt Accuracy/F1-macro 1.000 trên 30 ảnh test. Bài 2, đặc trưng hình học đạt 1.000 trên tập test; HOG-KNN có kết quả thấp hơn. Xem các CSV và confusion matrix trong `results/`.

Chạy `week6_solution.py` sẽ tạo `bao_cao_tuan_6.docx` trong thư mục này để chỉnh sửa cục bộ; `.gitignore` loại báo cáo khỏi các tệp Git. Nguồn và giấy phép dữ liệu được ghi trong `NOTICE.md`.
