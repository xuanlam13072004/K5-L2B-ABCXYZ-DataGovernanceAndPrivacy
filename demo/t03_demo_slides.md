# Demo T03 — Synthetic-vs-Real (4 slides)

## Slide 1 — Quyết định QC ảnh biến đổi

- **Vấn đề:** ảnh synthetic có thể hỏng usability hoặc làm nhãn bbox sai.
- **Mục tiêu:** phát hiện biến đổi không dùng được và giữ lại lineage để người phụ trách kiểm tra.
- **Luồng:** ảnh gốc → biến đổi → provenance → QC → review → GO / HOLD / REWORK.
- **Nói:** “Điểm cần kiểm soát là cả chất lượng ảnh lẫn việc nhãn còn đúng sau biến đổi.”

## Slide 2 — Contract dữ liệu và metric

- Fixture CC0 synthetic: 30 ảnh gốc (20 dev, 10 holdout), 45 ảnh holdout transformed; seed 42.
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
- Hai rủi ro cần đo trên heldout: erase cục bộ bị bỏ lọt; flare lệch tâm/clipping làm sai quyết định bbox.
- **Chưa có F1 baseline/PoC:** learner checkout không có GT overlay; generator dev hiện trả rỗng nên manifest provenance không được tạo.
- Không tuyên bố sim-to-real hoặc sẵn sàng production từ fixture synthetic.
- **Nói:** “Bước kế tiếp là hoàn thành generator/dev gold, nhận GT overlay từ giảng viên, sau đó chạy hai evaluator cùng seed và xuất bảng bằng `src.compare`.”

## Kịch bản phản biện nhanh

**Hỏi:** Vì sao chưa báo F1?  
**Đáp:** GT holdout là instructor-only và không có trong checkout; hai metrics JSON cũng chưa có. Tự suy ra nhãn từ ảnh sẽ làm benchmark không độc lập.

**Hỏi:** Ngưỡng đã tối ưu chưa?  
**Đáp:** Contract khóa ngưỡng hiện có trong config. Repo chưa có dev gold được sinh để xác nhận tuning; không tuyên bố ngưỡng tối ưu và không tune trên holdout.

**Hỏi:** Tại sao cần reviewer nếu có quality gate?  
**Đáp:** Rule ảnh có thể miss glare/erase ngoài các giả định đã mã hóa. Queue và audit lưu lại override, còn thiếu provenance thì HOLD.

**Hỏi:** Có thể đưa lên ảnh camera thật không?  
**Đáp:** Chưa. Chỉ có fixture synthetic; cần dataset thật có quyền sử dụng, protocol gán nhãn và đánh giá downstream trước khi cân nhắc.
