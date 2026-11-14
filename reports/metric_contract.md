# Metric contract — T03 Synthetic-vs-Real

**Trạng thái:** Bản khai của Member 6, ngày 2026-11-14. Contract được chốt trước lần đánh giá holdout tiếp theo; ngưỡng dưới đây lấy nguyên trạng từ config hiện có, chưa được xác nhận đã tune trên dev gold.

| Mục | Cam kết của nhóm |
|---|---|
| Track | T03 — Synthetic-vs-Real |
| Primary metric | **Transform QC F1** (`src/metrics/tracks.py::eval_t03`) |
| Hướng tốt | Cao hơn |
| Đơn vị | Một ảnh biến đổi, nối prediction với GT bằng `image_id` chính xác |
| Positive | `unusable = 1` theo `fixtures/t03_synthetic/README_usability_rule.md` |
| Công thức | TP: GT unusable và dự đoán 1; FP: GT usable, dự đoán 1; FN: GT unusable, dự đoán 0. Precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN). Khi mẫu số metric bằng 0, evaluator trả `None` nếu không xác định được metric. |
| Ghép prediction ↔ GT | Join chính xác trên `image_id`; ID lặp trong `qc.csv` làm evaluator thất bại. Prediction thiếu bị tính là usable (0), theo evaluator. |
| Candidate pool | Cố định: toàn bộ 45 ID trong `fixtures/t03_synthetic/holdout_transformed.csv`; không lọc theo score. |
| Review budget | B = 10 ảnh nghi ngờ cao nhất cho HITL; B không thay đổi primary metric. |
| Seed | 42 |
| Threshold policy | Dùng `params.thresholds` trong `configs/lab.yaml` (các giá trị mặc định tương ứng ở `configs/tracks/t03.yaml`); không điều chỉnh sau khi xem holdout. Tài liệu yêu cầu tuning trên dev gold, nhưng dev gold chưa có trong checkout nên hiện chưa xác nhận các ngưỡng là dev-tuned. |
| Eval split | `holdout_transformed` (10 ảnh gốc holdout, 45 biến thể). Không dùng split này để chọn ngưỡng. |
| Chỉ số lỗi xấu | Missed label-invalidating edits (đặc biệt `erase`); đồng thời báo precision/recall để thể hiện false alarm và false reject. |
| Slices | `transform` và `severity` theo GT; evaluator hiện xuất slice theo `transform`. Severity slice sẽ được bổ sung sau khi có GT overlay, không suy đoán từ tên ảnh. |
| Bất định khi N nhỏ | Báo khoảng tin cậy 95% bằng bootstrap theo cụm `parent_image` (2.000 lần lấy mẫu, seed 42) khi GT sẵn có; nhóm biến thể cùng ảnh gốc không được xem là độc lập. CI chưa tính trong checkout này. |
| Chốt / người chốt | 2026-11-14 · Member 6 (bản khai; phê duyệt nhóm chưa ghi nhận). |

## Định nghĩa unusable đã khóa

Ảnh là unusable nếu thỏa ít nhất một điều kiện: severity 2 của night/fog/blur/glare; glare severity ≥ 1 có tâm trong bbox đối tượng; hoặc erase xóa đối tượng nhưng nhãn vẫn còn. Đây là rule trong fixture, không phải kết luận về chất lượng dữ liệu ngoài đời.

## Điều kiện để tạo bảng benchmark

Chấm từng run bằng evaluator chuẩn rồi tạo bảng, không nhập số thủ công:

```bash
python -m src.evaluate --track t03 --pred out/baseline/qc.csv --run-name baseline --out reports/metrics_baseline.json
python -m src.evaluate --track t03 --pred out/poc/qc.csv --run-name poc --out reports/metrics_poc.json
python -m src.compare reports/metrics_baseline.json reports/metrics_poc.json --out reports/table1_baseline_vs_poc.md
```

Nếu baseline bằng 0, ghi `baseline=0` như `src.compare` quy định; không báo phần trăm cải thiện không xác định. Trong checkout hiện tại, GT overlay và output run chưa sẵn có; vì vậy chưa có benchmark score có thể tái lập.
