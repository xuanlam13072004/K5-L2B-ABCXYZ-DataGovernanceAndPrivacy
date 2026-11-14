# Báo cáo nhóm — Day 06 Data Lab · T03 Synthetic-vs-Real

> Benchmark nội bộ ngày 2026-11-14 dùng bbox GT trong data/obj_train_data.zip và ảnh data/image. Nhãn usability được sinh độc lập bằng tools.make_gold_set theo rule T03; điểm này không phải điểm hidden evaluation của giảng viên.

## 00 · Thông tin nhóm

| Mục | Giá trị |
|---|---|
| Group ID | G01 (configs/lab.yaml) |
| Challenge | T03 · Synthetic-vs-Real |
| Vai trò | M1 Integration · M2 Data/Provenance · M3 Generator/Baseline · M4 QC · M5 HITL · M6 QA/Benchmark/Report |
| Tên thật / Git URL / deadline cụ thể | Chưa khai báo trong tài liệu nhóm |
| Tag demo-180m / final-24h | Chưa có trong checkout |
| M6 | Git author maawongf; commit deliverables ban đầu e0b7c45; bảng đóng góp: team_contributions.csv |

## 01 · Bài toán và quyết định

Ảnh synthetic có thể hỏng usability hoặc làm nhãn đối tượng không còn đúng. Người gán nhãn và nhóm train mô hình chịu hậu quả nếu các ảnh này được đưa vào tập train. Pipeline nhận ảnh gốc, bbox và ảnh biến đổi; xuất QC theo image_id, queue review, audit và quyết định.

Chính sách M5 hiện tại: HOLD nếu thiếu/rỗng provenance; REWORK nếu tỷ lệ unusable dự đoán vượt 0.30; GO trong các trường hợp còn lại. Đây là chính sách PoC; usability ngoài đời cần domain owner xác nhận.

## 02 · Dữ liệu và provenance

| Bộ dữ liệu | Vai trò / kích thước | Nguồn và quyền sử dụng |
|---|---|---|
| Fixture chính thức | 30 ảnh gốc, 45 holdout transformed | Synthetic CC0 theo manifest; chưa có GT hidden overlay |
| Ảnh nhóm | 30 ảnh data/image, 1280×720; chia 20 dev / 10 holdout, seed 42 | Ảnh thực; nguồn/license chưa được nhóm xác nhận |
| Bbox GT | 30 file YOLO trong obj_train_data.zip; đủ 10 class names | File zip do Lead cung cấp |
| Input chuẩn hóa | 168 bbox giữ lại, 10 bbox bị loại theo importer hiện có | out/member6/source/label_fixes.csv ghi lý do |
| Gold nội bộ | 130 biến thể từ đúng 10 ảnh gốc holdout; 52 unusable, 78 usable | Night/glare/fog/blur × severity 0–2, cộng erase |
| Evaluation GT nội bộ | image_id, unusable, transform, severity, note | out/member6/gold/team_holdout_gt.csv, không đưa vào input pipeline |

ZIP là nhãn đối tượng của ảnh gốc. Công cụ import áp dụng mapping và cleanup của nhóm; các bbox bị loại chưa được Member 6 xác nhận bằng review thủ công. Gold usability dựa trên phép biến đổi đã biết, không suy đoán nhãn hidden fixture. Erase dùng inpainting theo bbox; nhãn unusable theo operation chưa chứng minh mọi đối tượng thực sự biến mất hoàn toàn.

Ảnh và GT sinh thêm nằm dưới out/ (gitignored). SHA256 của ZIP, GT và config được ghi trong t03_benchmark_details.json. Script không đưa ảnh thật vào reports/demo.

Lần kiểm tra fixture trước ghi nhận 14 manifest fail hash, riêng T03 lệch 5 file văn bản; Git blob khớp manifest nhưng working tree khác. CRLF/core.autocrlf=true là dấu hiệu nguyên nhân. Benchmark nội bộ dùng manifest mới tạo, strict provenance vẫn bật và cả hai run exit 0.

## 03 · Thiết kế

Luồng: import bbox → chọn 10 ảnh holdout bằng split cố định → tạo biến thể và GT bên ngoài pipeline → baseline/PoC cùng input → evaluator chuẩn → compare → báo cáo lỗi.

Baseline dùng Laplacian và độ tối toàn khung hình. PoC so với ảnh cha, đo sáng/tương phản/Laplacian, ước lượng tâm glare theo bbox và kiểm texture trong bbox để dò erase. Dùng nguyên ngưỡng config trước benchmark; không sửa thuật toán/ngưỡng sau khi xem kết quả.

Các phép biến đổi và quality gate đều chạy CPU offline, không dùng model pretrained. Dùng ảnh cha hỗ trợ chẩn đoán nhưng chỉ phù hợp khi ảnh cha và lineage sẵn có. Tiêu chí metric lấy từ Track T03, usability rule và eval_t03, xem [metric contract](metric_contract.md).

## 04 · Baseline vs PoC — benchmark nội bộ

Bảng dưới do src.compare sinh từ metrics_baseline.json và metrics_poc.json; hai run cùng 130 ảnh, seed 42, B=10, không reviewer override:

Baseline: `baseline` (git 69e2436, seed 42) · PoC: `poc` (git 69e2436, seed 42)

| Metric (unit) | Slice | Baseline | PoC | Δ abs | Δ rel | Better | N |
|---|---|---|---|---|---|---|---|
| **Transform QC F1** (F1, positive = unusable) | all | 0.5567 | 0.7619 | 0.2052 | +36.9% | higher | 130 |
| Precision (unusable) (TP/(TP+FP)) | all | 0.6000 | 1.0000 | 0.4000 | +66.7% | higher | 130 |
| Recall (unusable) (TP/(TP+FN)) | all | 0.5192 | 0.6154 | 0.0962 | +18.5% | higher | 130 |
| Missed label-invalidating edits (count) | all | 10.0000 | 10.0000 | 0.0000 | +0.0% | lower | None |
| Provenance completeness (manifest rows with all required fields) | all | n/a | n/a | n/a | n/a | higher | None |
| Runtime / 100 items (s) | all | 1.8314 | 21.8512 | | | lower | |

PoC tăng F1 nhưng vẫn bỏ lọt 20/52 positive. Chi phí xử lý tăng theo so sánh ảnh cha và bbox; runtime có trong run_info, không phải phép đo detector downstream. Provenance completeness là n/a vì baseline không xuất manifest và PoC xuất manifest rỗng; không được diễn giải thành 100%.

## 05 · Protocol và bất định

Join GT/prediction chính xác theo image_id, positive = unusable theo rule đã chốt. F1 = 2TP/(2TP+FP+FN), precision = TP/(TP+FP), recall = TP/(TP+FN); evaluator giữ giá trị không xác định là null. Không đổi pool/ngưỡng sau khi xem kết quả. 20 ảnh dev của nguồn không tham gia benchmark này.

Đã bootstrap 2.000 lần theo 10 cụm parent_image, seed 42: F1 CI95 baseline [0.520833, 0.591837], PoC [0.750000, 0.772727]. CI và transform/severity slices do tools.summarize_t03_benchmark sinh ở [chi tiết](t03_benchmark_details.md). Chỉ có 10 parent, phép sinh cùng family với thuật toán, và lựa chọn dữ liệu của nhóm; CI không bao phủ bias do chọn mẫu hoặc sim-to-real.

Repo đã có bộ gold real dùng ảnh nhóm trước lần chạy này. Chưa xác nhận các thành viên đã xem/tune trên những ảnh nào; tên holdout ở đây chỉ split của importer, không chứng minh bộ test hoàn toàn chưa từng được nhóm sử dụng.

## 06 · Hai failure cases đã xác nhận

Metadata và đặc trưng số nằm trong t03_benchmark_details.json; ảnh before/after giữ local trong out/member6/gold/team_holdout.

1. **g0011 — blur severity 2**, parent orig_05, nguồn data/image/6.jpg. GT=1, PoC=0, reason severity_1_blur_usable. Laplacian ratio=0.008397289540716184 vượt ngưỡng severity 2 là 0.0032; std ratio=0.9710903167724609 vượt ngưỡng 0.895. PoC hạ mức blur thành severity 1. Đây là ví dụ ngưỡng fixture không chuyển tốt sang ảnh thật độ phân giải cao; slice blur bỏ lọt 10/10 positive.
2. **g0012 — erase**, cùng parent và ảnh nguồn. GT=1, PoC=0, reason usable_clean. Global lap ratio=0.8975443300199364 cho phép bước dò erase, nhưng bbox sau inpainting còn std=34.73826599121094 và color_std=35.3848762512207, lớn hơn ngưỡng 3.5/6.0. Detector giả định vùng xóa gần đồng màu, trong khi inpainting giữ texture; cả 10 erase bị lọt.

Đây là root cause từ code và đặc trưng ảnh, không phải kết luận threshold mới cần dùng. Bước sửa thuộc M4, cần xác nhận trên dev độc lập rồi chạy heldout mới.

**Sim-to-real gap:** benchmark đã dùng ảnh gốc thật nhưng weather/corruption vẫn synthetic. Chưa có GT thời tiết tự nhiên, nhãn downstream hay detector để đo gap định lượng. Exposure, lens flare nhiều đỉnh, sensor noise, chuyển động và temporal correlation của camera thật chưa được mô phỏng đầy đủ.

## 07 · Human review và quyết định

PoC tạo contact sheet before/after và queue top-10 ở out/member6/poc. Chưa có reviewer override cho run này; n_reviewed=0. Audit/decision giữ local cùng run.

Quyết định hiện tại là HOLD / MISSING_PROVENANCE_MANIFEST: n_generated=0 do src/t03/generate.py còn TODO. Tỷ lệ unusable dự đoán 0.2462 thấp hơn 0.30 cũng không đủ cho release. Provenance phục vụ tạo GT benchmark nằm riêng ở gold/team_holdout_provenance.csv; đó không phải manifest do PoC xuất.

## 08 · Tái lập và tests

Từ checkout sạch với data/image/*.jpg và data/obj_train_data.zip đã có, chạy:

```powershell
python -m tools.import_team_labels --out out/member6/source --seed 42
python -m tools.make_gold_set --source out/member6/source --split holdout --name team_holdout --out-root out/member6/gold --seed 42
python -m src.main --track t03 --input out/member6/gold/team_holdout --pipeline src.pipeline_baseline --output out/member6/baseline --run-name baseline --offline --seed 42
python -m src.main --track t03 --input out/member6/gold/team_holdout --output out/member6/poc --run-name poc --offline --seed 42
python -m src.evaluate --track t03 --gt out/member6/gold/team_holdout_gt.csv --pred out/member6/baseline/qc.csv --run-name baseline --out reports/metrics_baseline.json
python -m src.evaluate --track t03 --gt out/member6/gold/team_holdout_gt.csv --pred out/member6/poc/qc.csv --run-name poc --out reports/metrics_poc.json
python -X utf8 -m src.compare reports/metrics_baseline.json reports/metrics_poc.json --out reports/table1_baseline_vs_poc.md
python -m tools.summarize_t03_benchmark
python -m pytest -q tests/test_t03.py
```

Windows/Python 3.13.14. Dùng -X utf8 cho compare vì console cp1258 không in được ký tự delta. Các công cụ import/gold thay thế thư mục output sinh trước đó, vì vậy dùng đúng các đường dẫn out/member6 ở trên.

Tests M6 pass 4/4. Lần full-suite trước có 17 lỗi (14 manifest fixture + 3 E2E T03 vì strict provenance). Benchmark nội bộ exit 0 không khắc phục các lỗi fixture đó.

## 09 · Giới hạn và cải tiến

- Xác nhận lại license/source của ảnh thật và review bbox cleanup trước chia sẻ hay release.
- Đánh giá trên nhãn usability thủ công độc lập để kiểm chứng nhãn suy từ operation; không xem bbox zip là nhãn thời tiết tự nhiên.
- M4 xử lý miss blur/erase trên dev; M3/M2 hoàn thiện generator/manifest; giữ threshold heldout cố định.
- Bổ sung human review có người chịu trách nhiệm và audit; hiện mới có queue, chưa có xác nhận của reviewer cho benchmark.
- Không suy từ các số này thành đảm bảo chất lượng production hoặc tuân thủ pháp luật.

## 10 · Đóng góp và tài liệu

Bảng [team_contributions.csv](team_contributions.csv) ghi đóng góp Member 6. Các thành viên khác chưa có ánh xạ tên thật/email đầy đủ nên không tự gán tác giả.

Tài liệu dùng: docs/tracks/T03.md, fixtures/t03_synthetic/README_usability_rule.md, src/metrics/tracks.py::eval_t03, tools/import_team_labels.py và tools/make_gold_set.py. Các metrics/table/details trong reports được sinh bằng script, không nhập số thay evaluator.
