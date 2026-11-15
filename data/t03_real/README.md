# data/t03_real — 30 ảnh của nhóm G01 ở định dạng pipeline T03

Thư mục này được **tạo tự động** từ `data/image/*.jpg` + `data/obj_train_data.zip` bằng
`python -m tools.import_team_labels` (seed 42). `python -m src.main` tự chạy lệnh đó nếu chưa có `originals.csv`.
Chỉ file README này được commit; mọi file khác là sản phẩm sinh ra (xem `.gitignore`).

| File | Nội dung |
|---|---|
| `originals/`, `originals.csv` | 30 ảnh gốc `orig_00…orig_29`, chia 20 dev / 10 holdout |
| `objects.csv` | bbox đã làm sạch, 4 lớp `car / pedestrian / sign` (cột `orig_class` giữ lớp YOLO gốc) |
| `label_fixes.csv` | các bbox bị loại và lý do |
| `holdout_transformed/`, `holdout_transformed.csv` | 45 ảnh biến đổi từ 10 ảnh holdout (không kèm đáp án) |
| `source_map.csv`, `manifest.csv` | nguồn từng ảnh, sha256 từng file |
