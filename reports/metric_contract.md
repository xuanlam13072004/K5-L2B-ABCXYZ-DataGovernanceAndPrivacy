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
| Candidate pool | Pipeline mặc định: 45 ID trong `data/t03_real/holdout_transformed.csv` (ảnh nhóm, không kèm đáp án); fixture giảng viên chỉ để tham khảo. Benchmark nội bộ: 130 ID trong `out/member6/gold/team_holdout/holdout_transformed.csv`, từ 10 ảnh holdout nguồn nhóm; không lọc theo score. |
| Review budget | B = 10 ảnh nghi ngờ cao nhất cho HITL; B không thay đổi primary metric. |
| Seed | 42 |
| Threshold policy | v1 (14/11): ngưỡng của M4, chưa tune trên dev độc lập. v2 (15/11): 4 ngưỡng mới (`blur_sev1_lap_ratio` 0.30, `blur_sev2_radius_px` 3.5, `erase_min_inside_mad` 8.0, `erase_min_change_ratio` 20.0) chỉ chọn trên team_dev = 20 ảnh dev × 13 biến thể (`tools.make_gold_set --split dev`), ghi vào `configs/lab.yaml` trước khi chạy heldout. Heldout đã được xem ở v1 (biết loại lỗi), xem report mục 05. |
| Eval split | Nội bộ: importer chia 30 ảnh nhóm thành 20 dev/10 holdout bằng seed 42; chỉ 10 holdout sinh 130 biến thể. Fixture giảng viên chỉ để tham khảo. Không dùng heldout chọn ngưỡng. |
| Chỉ số lỗi xấu | Missed label-invalidating edits (đặc biệt `erase`); đồng thời báo precision/recall để thể hiện false alarm và false reject. |
| Slices | `transform` và `severity` theo GT sinh độc lập, ghi trong `reports/t03_benchmark_details.md`; evaluator chuẩn xuất transform slices. |
| Bất định khi N nhỏ | Đã bootstrap theo cụm `parent_image`, 2.000 lần, seed 42, CI95 trong chi tiết benchmark. Biến thể cùng parent không xem là độc lập; CI không đo bias chọn mẫu hoặc sim-to-real. |
| Chốt / người chốt | 2026-11-14 · Member 6 (bản khai; phê duyệt nhóm chưa ghi nhận). |

## Định nghĩa unusable đã khóa

Ảnh là unusable nếu thỏa ít nhất một điều kiện: severity 2 của night/fog/blur/glare; glare severity ≥ 1 có tâm trong bbox đối tượng; hoặc erase xóa đối tượng nhưng nhãn vẫn còn. Đây là rule trong fixture, không phải kết luận về chất lượng dữ liệu ngoài đời.

## Điều kiện để tạo bảng benchmark

Chấm từng run bằng evaluator chuẩn rồi tạo bảng, không nhập số thủ công:

```bash
python -m src.evaluate --track t03 --gt out/member6/gold/team_holdout_gt.csv --pred out/member6/baseline/qc.csv --run-name baseline --out reports/metrics_baseline.json
python -m src.evaluate --track t03 --gt out/member6/gold/team_holdout_gt.csv --pred out/member6/poc/qc.csv --run-name poc --out reports/metrics_poc.json
python -X utf8 -m src.compare reports/metrics_baseline.json reports/metrics_poc.json --out reports/table1_baseline_vs_poc.md
python -m tools.summarize_t03_benchmark
```

Nếu baseline bằng 0, ghi `baseline=0` như `src.compare` quy định. Benchmark nội bộ đã chạy từ bbox GT trong ZIP và công cụ `tools.make_gold_set`; xem lệnh dựng dữ liệu ở `group_report.md`. Nhãn usability suy từ phép biến đổi, không phải nhãn hidden fixture hay GT thời tiết tự nhiên. GT nằm ngoài thư mục input pipeline; pipeline không đọc nó. Các ngưỡng giữ nguyên trước và sau benchmark.
