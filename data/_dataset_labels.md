# Dataset labels — T03

Đối chiếu trực tiếp data/obj_train_data.zip ngày 2026-11-14: obj.data khai báo 10 classes, obj.names có đủ 10 tên. ZIP có 30 file annotation YOLO, train.txt, obj.data và obj.names; không chứa ảnh. Ảnh nguồn tương ứng nằm trong data/image.

| Class ID | Tên trong obj.names |
|---:|---|
| 0 | bus |
| 1 | car |
| 2 | fire hydrant |
| 3 | person |
| 4 | stop sign |
| 5 | tennis racket |
| 6 | traffic light |
| 7 | train |
| 8 | truck |
| 9 | potted plant |

Mỗi dòng YOLO: class_id x_center y_center width height, tọa độ chuẩn hóa theo kích thước ảnh. Class 9 được định nghĩa là potted plant; ghi chú cũ về thiếu tên lớp không còn đúng với ZIP hiện tại.

Đây là GT bbox của ảnh gốc, chưa phải nhãn unusable của ảnh biến đổi. tools.import_team_labels ánh xạ bus/car/truck → car, person → pedestrian, stop sign/traffic light → sign; loại các lớp ngoài taxonomy, bbox ego hood và duplicate theo rule hiện có. Việc nằm ngoài taxonomy không tự chứng minh nhãn sai. Danh sách bị loại nằm ở label_fixes.csv, cần nhóm review.

Benchmark Member 6 dùng split holdout của nguồn nhóm để tạo gold usability theo rule T03 bằng tools.make_gold_set. Xem reports/group_report.md và t03_benchmark_details.json cho protocol và SHA256 ZIP. Ảnh thực, source/license và trạng thái review bbox cần được nhóm xác nhận riêng.
