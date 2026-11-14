# Demo T03 — Synthetic-vs-Real (4 slides)

## Slide 1 — Quyết định QC ảnh biến đổi

- **Vấn đề:** ảnh synthetic có thể hỏng usability hoặc làm nhãn bbox sai.
- **Mục tiêu:** phát hiện biến đổi không dùng được và giữ lại lineage để người phụ trách kiểm tra.
- **Luồng:** ảnh gốc → biến đổi → provenance → QC → review → GO / HOLD / REWORK.
- **Nói:** “Điểm cần kiểm soát là cả chất lượng ảnh lẫn việc nhãn còn đúng sau biến đổi.”

## Slide 2 — Contract dữ liệu và metric

- Nội bộ: 30 ảnh thật + bbox YOLO từ ZIP; chia 20 dev/10 holdout, sinh 130 biến thể từ 10 holdout; seed 42. GT fixture chính thức vẫn chưa có.
- Positive = unusable theo rule đã chốt: severity 2, glare trên bbox, hoặc erase nhưng nhãn còn.
- Primary: Transform QC F1, join chính xác bằng `image_id`; báo precision, recall và lỗi erase.
- **Nói:** “Không lấy ảnh biến thể cùng parent làm các mẫu độc lập khi tính bất định.”

## Slide 3 — Quality gate và human review

- PoC so ảnh với ảnh cha; đo sáng, tương phản, Laplacian; dò glare theo bbox và vùng erase.
- Baseline chỉ dùng Laplacian/độ tối toàn ảnh.
- M5 ưu tiên top-10, contact sheet offline, nhận override và ghi audit log.
- **Nói:** “QC tự động đưa ra tín hiệu; HOLD khi thiếu provenance giúp tránh release ảnh không truy xuất được.”

## Slide 4 — Bằng chứng, giới hạn và bước kế tiếp

- Tests M6: 4 test file `tests/test_t03.py` pass; full suite hiện có 17 failure do hash manifest và strict provenance.
- **F1 nội bộ:** baseline 0.5567 → PoC 0.7619 (130 ảnh); bảng do src.compare sinh.
- PoC bỏ lọt 10 blur severity 2 và 10 erase; ví dụ g0011/g0012 trên parent orig_05. Ngưỡng blur quá chặt; inpainting không tạo bbox đồng màu như detector giả định.
- Pipeline **HOLD** vì manifest provenance rỗng; chưa có human review hay đánh giá thời tiết thật.
- **Nói:** “Bước kế tiếp là xử lý lỗi trên dev độc lập, hoàn thiện provenance và đánh giá mới; không đổi ngưỡng trên chính bộ heldout vừa chấm.”

## Kịch bản phản biện nhanh

**Hỏi:** Đây có phải điểm hidden eval chính thức không?

**Đáp:** Đây là benchmark nội bộ: bbox GT của nhóm kết hợp phép biến đổi biết trước sinh nhãn usability. GT hidden fixture vẫn do giảng viên giữ; hai bộ không được gộp.

**Hỏi:** Ngưỡng đã tối ưu chưa?  
**Đáp:** Ngưỡng config giữ nguyên trước và sau benchmark; kết quả blur/erase cho thấy cần thử lại trên dev độc lập. Không dùng bộ vừa chấm để tune rồi báo lại như holdout mới.

**Hỏi:** Tại sao cần reviewer nếu có quality gate?  
**Đáp:** Rule ảnh có thể miss glare/erase ngoài các giả định đã mã hóa. Queue và audit lưu lại override, còn thiếu provenance thì HOLD.

**Hỏi:** Có thể đưa lên ảnh camera thật không?  
**Đáp:** Đã chạy trên ảnh gốc thật nhưng nhiễu/thời tiết vẫn synthetic. Cần xác nhận license, review bbox và kiểm trên thời tiết tự nhiên/downstream trước khi cân nhắc triển khai.
