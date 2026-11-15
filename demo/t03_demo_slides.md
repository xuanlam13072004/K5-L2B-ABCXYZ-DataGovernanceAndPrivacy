# Demo T03 — Synthetic-vs-Real (4 slides)

## Slide 1 — Quyết định QC ảnh biến đổi

- **Vấn đề:** ảnh synthetic có thể hỏng usability hoặc làm nhãn bbox sai.
- **Mục tiêu:** phát hiện biến đổi không dùng được và giữ lại lineage để người phụ trách kiểm tra.
- **Luồng:** ảnh gốc → biến đổi → provenance → QC → review → GO / HOLD / REWORK.
- **Nói:** “Điểm cần kiểm soát là cả chất lượng ảnh lẫn việc nhãn còn đúng sau biến đổi.”

## Slide 2 — Contract dữ liệu và metric

- Nội bộ: 30 ảnh thật + bbox YOLO từ ZIP; chia 20 dev/10 holdout, sinh 130 biến thể từ 10 holdout; seed 42. Ảnh: 30 frame BDD100K (chỉ nghiên cứu/học tập).
- Positive = unusable theo rule đã chốt: severity 2, glare trên bbox, hoặc erase nhưng nhãn còn.
- Primary: Transform QC F1, join chính xác bằng `image_id`; báo precision, recall và lỗi erase.
- **Nói:** “Không lấy ảnh biến thể cùng parent làm các mẫu độc lập khi tính bất định.”

## Slide 3 — Quality gate và human review

- Pipeline tự sinh biến thể từ 20 ảnh dev và ghi `transform_manifest.csv` (seed, tâm glare, sha256 cha/con): provenance 1.0.
- PoC v2 so ảnh với ảnh cha: sáng/tương phản, ước lượng bán kính blur, tâm glare so với bbox, thay đổi tập trung trong bbox (erase).
- Baseline chỉ dùng Laplacian/độ tối toàn ảnh.
- M5 ưu tiên top-10, contact sheet offline, nhận override và ghi audit log.
- **Nói:** “QC tự động đưa ra tín hiệu; HOLD khi thiếu provenance giúp tránh release ảnh không truy xuất được.”

## Slide 4 — Bằng chứng, giới hạn và bước kế tiếp

- `python -m pytest -q`: 55 passed, 0 failed trên checkout sạch; một lệnh `python -m src.main ... --offline --seed 42` chạy hết 5 bước.
- **F1 nội bộ (130 ảnh heldout):** baseline 0.5567 → PoC v1 0.7619 → **PoC v2 1.0000**; bảng do src.compare sinh, ablation cho thấy mỗi kiểm tra đều đóng góp.
- Hai failure mode bắt buộc đã gặp thật và sửa: erase bằng inpainting bị lọt 10/10 (v1); manifest provenance rỗng → HOLD. Sau sửa: 0 erase lọt, completeness 1.0.
- Quyết định chạy mặc định: **REWORK** — 25/45 ảnh holdout unusable (55.6% > 30%): lô biến thể quá nặng, cần sinh lại nhẹ hơn.
- **Nói:** “F1 = 1.0 là trên biến đổi synthetic do chính nhóm sinh; nó chưa chứng minh gì với thời tiết thật. Bước tiếp theo là thử motion blur, lens flare, sương mù thật.”

## Kịch bản phản biện nhanh

**Hỏi:** Đây có phải điểm hidden eval chính thức không?

**Đáp:** Đây là benchmark nội bộ: bbox GT của nhóm kết hợp phép biến đổi biết trước sinh nhãn usability. GT hidden fixture vẫn do giảng viên giữ; hai bộ không được gộp.

**Hỏi:** Ngưỡng đã tối ưu chưa?  
**Đáp:** Ngưỡng v2 chỉ chọn trên 20 ảnh dev và ghi vào config trước khi chạy heldout. Heldout đã được xem ở v1 nên nhóm nói rõ điều đó trong report; không coi F1 = 1.0 là bảo đảm.

**Hỏi:** Tại sao cần reviewer nếu có quality gate?  
**Đáp:** Rule ảnh có thể miss glare/erase ngoài các giả định đã mã hóa. Queue và audit lưu lại override, còn thiếu provenance thì HOLD.

**Hỏi:** Có thể đưa lên ảnh camera thật không?  
**Đáp:** Đã chạy trên ảnh gốc thật nhưng nhiễu/thời tiết vẫn synthetic. Cần review bbox, kiểm trên thời tiết tự nhiên/downstream trước khi cân nhắc triển khai; BDD100K không cho dùng thương mại.
