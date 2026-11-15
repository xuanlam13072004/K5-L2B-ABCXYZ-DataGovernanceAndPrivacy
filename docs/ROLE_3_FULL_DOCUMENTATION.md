# BÁO CÁO KỸ THUẬT VÀ TÀI LIỆU ROLE 3 (T03 - SYNTHETIC VS REAL)

**Người thực hiện**: Role 3 – Synthetic Generator & Baseline Engineer  
**Mục tiêu**:
1. Xây dựng bộ sinh ảnh giả lập/biến đổi (Synthetic Data Generator) từ dữ liệu thực tế `data/image/` mô phỏng suy giảm cảm biến xe tự hành.
2. Xây dựng Pipeline Baseline gọn nhẹ (Heuristic rule-based) để thiết lập chuẩn so sánh (benchmark threshold) trước khi Role 4 phát triển thuật toán PoC nâng cao.

---

## 1. TỔNG QUAN WORKFLOW VÀ VỊ TRÍ CỦA ROLE 3

```
+-------------------------------------------------------------------------------+
|                                  DATA PIPELINE                                |
+-------------------------------------------------------------------------------+
               |
               v
      [30 Ảnh data/image/]
         |           |
         |           +---------------------------------------+
         v                                                   v
 [Role 3: tools/auto_label_for_cvat.py]       [Role 3: tools/generate_synthetic_dev.py]
         | (Pre-annotation YOLOv8/11)                        | (Physics-based Corruptions)
         v                                                   v
 [data/cvat_export/cvat_yolo_upload.zip]      [240 Ảnh data/custom_synthetic/]
         |                                                   |
         v (Chuyển giao)                                     +-------------------+
 [Role 2: Review nhãn trên CVAT]                                                 |
         | (Sửa nhãn, viết Data Dictionary)                                      v
         v                                                   [Role 3: src/pipeline_baseline.py]
 [Ground Truth sạch: labels.csv/json]                                | (Heuristic Laplacian + Brightness)
         |                                                           v
         +-------------------------------------------------> [out/baseline/qc.csv]
                                                             [out/baseline/transform_manifest.csv]
                                                                     |
                                                                     v (Bàn giao Benchmark)
                                                             [Role 4: Phát triển PoC Pipeline]
                                                             [Role 6: QA Benchmark So sánh PoC vs Baseline]
```

### Giải đáp phụ thuộc: Role 3 có phải đợi Người thứ 2 (Role 2) duyệt nhãn không?
**KHÔNG.** Role 3 chạy độc lập 100% vì:
1. **Thuật toán sinh ảnh (`generate_synthetic_dev.py`)**: Hoạt động trực tiếp trên ma trận điểm ảnh RGB của ảnh gốc (Pixel-level manipulation). Quá trình thêm nhiễu (Night, Glare, Fog, Blur) không cần và không đụng tới Bounding Box.
2. **Pipeline Baseline (`pipeline_baseline.py`)**: Đánh giá độ sắc nét (Laplacian blur variance) và độ rọi (Luminance) trên toàn bộ khung hình ảnh.
3. **Ai mới cần đợi Người thứ 2?**: **Role 4 (Core Algorithm)** và **Role 5 (HITL Review)** mới cần file Ground Truth sau khi Người thứ 2 đã kiểm tra nhãn để đánh giá độ chính xác (mAP / F1 drop) và vẽ Contact Sheet kiểm duyệt.

---

## 2. THUẬT TOÁN SINH ẢNH BIẾN ĐỔI (SYNTHETIC DATA GENERATOR)

File thực thi: `tools/generate_synthetic_dev.py`  
Mô phỏng 4 điều kiện thời tiết/cảm biến thực tế của xe tự hành với 2 cấp độ nghiêm trọng (Severity 1 và Severity 2):

| Transform | Cơ chế vật lý | Mức s1 | Mức s2 |
| :--- | :--- | :--- | :--- |
| **night** (Ban đêm/thiếu sáng) | Giảm hệ số sáng RGB kênh phi tuyến tính: $I_{out} = I \times \gamma$ | $\gamma = 0.35$ | $\gamma = 0.15$ |
| **glare** (Lóa sáng ngược nắng/đèn pha) | Cộng điểm sáng Gaussian $I + s\cdot 255\cdot e^{-\frac{r^2}{2(R/2)^2}}$, $R = 0.35\cdot\max(H,W)$ (`corruptions.glare`). **Tâm lóa không cố định ở giữa ảnh:** trong pipeline (`src/t03/generate.py`) tâm được chọn có chủ đích, một nửa rơi vào bbox đối tượng, một nửa ngoài mọi bbox, và ghi vào cột `params` của `transform_manifest.csv` (nếu không truyền `center`, `corruptions.glare` chọn ngẫu nhiên) | $s = 0.6$ | $s = 1.0$ (bão hòa trắng quanh tâm) |
| **fog** (Sương mù khí quyển) | Mô hình tán xạ khí quyển Koschmieder: $I_{out} = I \cdot t + L(1 - t)$ với độ truyền qua $t$ | $t = 0.55$, sương nhẹ | $t = 0.25$, sương mù dày đặc |
| **blur** (Rung lắc camera/nhòe chuyển động) | Tích chập ma trận làm mờ Gaussian (Kernel $k \times k$, độ lệch $\sigma$) | Kernel $7 \times 7$, $\sigma = 1.5$ | Kernel $15 \times 15$, $\sigma = 3.5$ |

### Nguyên tắc Data Governance & Lineage:
- **Seed cố định**: `--seed 42` bảo đảm tính tất định (Reproducibility).
- **Truy xuất nguồn gốc (Lineage)**: Mỗi ảnh sinh ra đều được băm SHA-256 mã gốc và mã sinh, ghi nhận vào manifest:
  - Vị trí ảnh sinh: `data/custom_synthetic/synth_orig_{id}_{transform}_{severity}_{index}.jpg`
  - Vị trí manifest: `out/baseline/transform_manifest.csv` (240 dòng dữ liệu).

---

## 3. THIẾT KẾ PIPELINE BASELINE (HEURISTIC QUALITY GATE)

File thực thi: `src/pipeline_baseline.py` (~65 dòng code chuẩn mực, tuân thủ hợp đồng Lab).

### Cơ chế đánh giá Heuristic:
- **Độ nhòe (Blur Metric)**:
  $$\text{Var}(\nabla^2 I) = \text{Variance}(\text{Laplacian}(I_{\text{gray}}))$$
  Nếu $\text{Var} < 100.0 \implies$ Ảnh bị nhòe (FAIL).
- **Độ sáng (Luminance Metric)**:
  $$\bar{Y} = \frac{1}{HW}\sum_{x,y} (0.299R + 0.587G + 0.114B)$$
  Nếu $\bar{Y} < 40.0$ (quá tối) hoặc $\bar{Y} > 220.0$ (quá chói) $\implies$ Ảnh hỏng sáng (FAIL).
- **Trạng thái quyết định**: Toàn bộ dự đoán baseline gắn trạng thái `status: HOLD`, `reason_code: BASELINE_ONLY` để chuyển tuyến về bộ phận kiểm tra chất lượng HITL.

### File đầu ra của Baseline:
- `out/baseline/qc.csv`: Chứa danh sách toàn bộ ảnh kèm điểm `laplacian_var`, `mean_luminance`, nhãn dự đoán `predicted_pass`.
- `out/baseline/review_queue.csv`: Danh sách các ảnh bị đánh trượt cần kiểm duyệt thủ công.
- `out/baseline/decision.json`: Báo cáo thẩm định tự động.
- `out/baseline/audit_log.jsonl`: Nhật ký kiểm toán toàn vẹn dữ liệu.

---

## 4. HƯỚNG DẪN KIỂM TRA VÀ CHẠY THỰC TẾ

### 1. Sinh lại 240 ảnh synthetic:
```powershell
python tools/generate_synthetic_dev.py --input-dir data/image --output-dir data/custom_synthetic --seed 42
```

### 2. Chạy Pipeline Baseline:
```powershell
python -m src.pipeline_baseline --input-dir data/image --out-dir out/baseline
```

### 3. Công cụ hỗ trợ gán nhãn tự động (đã xuất sẵn cho Người thứ 2):
```powershell
# File zip đã chuẩn hóa chuẩn format YOLO 1.1 có obj.data sẵn sàng upload lên CVAT:
data/cvat_export/cvat_yolo_upload.zip
```

---

## 5. CHECKLIST BÀN GIAO CHO CÁC ROLE TIẾP THEO

- [x] **Bàn giao Role 2**: File `data/cvat_export/cvat_yolo_upload.zip` để Role 2 đưa lên CVAT duyệt nhãn, viết Data Dictionary.
- [x] **Bàn giao Role 4**: Bộ 240 ảnh `data/custom_synthetic/` và file kết quả `out/baseline/qc.csv` làm ngưỡng tiêu chuẩn cần vượt qua.
- [x] **Bàn giao Role 6**: File `src/pipeline_baseline.py` và `out/baseline/` để viết test suite so sánh mAP / F1-score giữa PoC và Baseline.
