# Báo cáo nhóm — Day 06 Data Lab · T03 Synthetic-vs-Real

> Benchmark nội bộ ngày 2026-11-14 dùng bbox GT trong data/obj_train_data.zip và ảnh data/image. Nhãn usability được sinh độc lập bằng tools.make_gold_set theo rule T03; điểm này không phải điểm hidden evaluation của giảng viên.

## 00 · Thông tin nhóm

| Mục | Giá trị |
|---|---|
| Group ID | G01 (configs/lab.yaml) |
| Challenge | T03 · Synthetic-vs-Real |
| Vai trò | M1 Integration · M2 Data/Provenance · M3 Generator/Baseline · M4 QC · M5 HITL · M6 QA/Benchmark/Report |
| Git URL | https://github.com/xuanlam13072004/K5-L2B-XuanLam-DataGovernanceAndPrivacy |
| Tên thật / deadline cụ thể | TODO (nhóm điền) |
| Tag demo-180m / final-24h | Chưa có trong checkout |
| M6 | Git author maawongf; commit deliverables ban đầu e0b7c45; bảng đóng góp: team_contributions.csv |

## 01 · Bài toán và quyết định

Ảnh synthetic có thể hỏng usability hoặc làm nhãn đối tượng không còn đúng. Người gán nhãn và nhóm train mô hình chịu hậu quả nếu các ảnh này được đưa vào tập train. Pipeline nhận ảnh gốc, bbox và ảnh biến đổi; xuất QC theo image_id, queue review, audit và quyết định.

Chính sách M5 hiện tại: HOLD nếu thiếu/rỗng provenance; REWORK nếu tỷ lệ unusable dự đoán vượt 0.30; GO trong các trường hợp còn lại. Đây là chính sách PoC; usability ngoài đời cần domain owner xác nhận.

## 02 · Dữ liệu và provenance

| Bộ dữ liệu | Vai trò / kích thước | Nguồn và quyền sử dụng |
|---|---|---|
| Fixture mẫu của giảng viên | 30 ảnh gốc, 45 holdout transformed; chỉ để tham khảo, nhóm không dùng để đánh giá | Synthetic CC0 theo manifest |
| Ảnh nhóm | 30 ảnh data/image, 1280×720; chia 20 dev / 10 holdout, seed 42 | 30 frame dashcam chọn từ BDD100K; BDD100K License, chỉ dùng nghiên cứu/học tập, không thương mại (https://doc.bdd100k.com/license.html). Có người và biển số thật: repo để private, không đưa ảnh vào demo công khai |
| Bbox GT | 30 file YOLO trong obj_train_data.zip; đủ 10 class names | File zip do Lead cung cấp |
| Input chuẩn hóa (input mặc định của pipeline) | data/t03_real: 168 bbox giữ lại, 10 bbox bị loại; 45 ảnh holdout_transformed không kèm đáp án | data/t03_real/label_fixes.csv ghi lý do; pipeline tự tạo thư mục này nếu thiếu |
| Gold nội bộ | 130 biến thể từ đúng 10 ảnh gốc holdout; 52 unusable, 78 usable | Night/glare/fog/blur × severity 0–2, cộng erase |
| Evaluation GT nội bộ | image_id, unusable, transform, severity, note | out/member6/gold/team_holdout_gt.csv, không đưa vào input pipeline |

ZIP là nhãn đối tượng của ảnh gốc. Công cụ import áp dụng mapping và cleanup của nhóm; các bbox bị loại chưa được Member 6 xác nhận bằng review thủ công. Gold usability dựa trên phép biến đổi đã biết, không suy đoán nhãn hidden fixture. Erase dùng inpainting theo bbox; nhãn unusable theo operation chưa chứng minh mọi đối tượng thực sự biến mất hoàn toàn.

Ảnh và GT sinh thêm nằm dưới out/ (gitignored). SHA256 của ZIP, GT và config được ghi trong t03_benchmark_details.json. Script không đưa ảnh thật vào reports/demo.

Lỗi hash fixture trước đây (14 manifest fail) do Git trên Windows đổi xuống dòng (core.autocrlf=true) trong working tree. Đã khắc phục bằng commit afd97ed (khôi phục manifest gốc) và 62627a8 (`.gitattributes`: `fixtures/** -text`). Trên checkout sạch `python -m tools.verify_manifests` cho 14 manifests, 0 failing; máy Windows clone trước đó cần xoá `fixtures/` rồi `git checkout -- fixtures`.

**Provenance của ảnh synthetic do PoC sinh:** bước 02 của pipeline (`src/t03/generate.py`) sinh 13 biến thể cho mỗi ảnh dev (night/glare/fog/blur × severity 0–2, cộng 1 erase bằng inpainting). `src/t03/provenance.py` ghi `transform_manifest.csv` với image_id, parent_image, transform, severity, seed, params (tâm glare, obj bị xoá), parent_sha256, sha256 ảnh con, generator, version; và `dev_gold.csv` gán nhãn theo usability rule. Chạy mặc định trên 30 ảnh: 260 ảnh sinh từ 20 ảnh dev, Provenance completeness = 1.0; chạy lại cùng seed cho cùng sha256.

## 03 · Thiết kế

Luồng pipeline (một lệnh `python -m src.main --config configs/lab.yaml --offline --seed 42`): load + kiểm sha256 → sinh biến thể dev có provenance → quality gate trên 45 ảnh holdout → rank top-10 + contact sheet → review override + audit log → quyết định GO/HOLD/REWORK.

Luồng benchmark: import bbox → chọn 10 ảnh holdout bằng split cố định → tạo biến thể và GT bên ngoài pipeline (`tools.make_gold_set`) → baseline/PoC cùng input → evaluator chuẩn → compare → ablation → báo cáo lỗi.

Baseline dùng Laplacian và độ tối toàn khung hình. PoC so với ảnh cha, đo sáng/tương phản/Laplacian, ước lượng tâm glare theo bbox và kiểm texture trong bbox để dò erase. Dùng nguyên ngưỡng config trước benchmark; không sửa thuật toán/ngưỡng sau khi xem kết quả.

**Quality gate v2** (`src/t03/qc.py`, 15/11): (a) mức blur đọc từ bán kính Gaussian ước lượng bằng cách làm mờ ảnh cha ở các bán kính ứng viên và chọn bán kính khớp nhất (MSE), thay cho tỉ lệ Laplacian vốn chồng lấn giữa severity 1 và 2 ở 1280×720 (0.010–0.076 so với 0.003–0.027); (b) erase còn được bắt khi thay đổi tập trung trong một bbox (mean |diff| trong bbox ≥ 8 và ≥ 20 lần phần ngoài bbox), không chỉ khi bbox thành mảng màu phẳng. Ngưỡng mới chỉ chọn trên team_dev (20 ảnh dev × 13 biến thể), ghi vào `configs/lab.yaml` trước khi chạy heldout.

Các phép biến đổi và quality gate đều chạy CPU offline, không dùng model pretrained. Dùng ảnh cha hỗ trợ chẩn đoán nhưng chỉ phù hợp khi ảnh cha và lineage sẵn có. Tiêu chí metric lấy từ Track T03, usability rule và eval_t03, xem [metric contract](metric_contract.md).

## 04 · Baseline vs PoC — benchmark nội bộ

Bảng dưới do src.compare sinh từ metrics_baseline.json và metrics_poc.json; hai run cùng 130 ảnh, seed 42, B=10, không reviewer override:

Baseline: `baseline` (git debf3cf, seed 42) · PoC: `poc` (git debf3cf, seed 42)

| Metric (unit) | Slice | Baseline | PoC | Δ abs | Δ rel | Better | N |
|---|---|---|---|---|---|---|---|
| **Transform QC F1** (F1, positive = unusable) | all | 0.5567 | 1.0000 | 0.4433 | +79.6% | higher | 130 |
| Precision (unusable) (TP/(TP+FP)) | all | 0.6000 | 1.0000 | 0.4000 | +66.7% | higher | 130 |
| Recall (unusable) (TP/(TP+FN)) | all | 0.5192 | 1.0000 | 0.4808 | +92.6% | higher | 130 |
| Missed label-invalidating edits (count) | all | 10.0000 | 0.0000 | -10.0000 | -100.0% | lower | None |
| Provenance completeness (manifest rows with all required fields) | all | n/a | 1.0000 | n/a | n/a | higher | None |
| Runtime / 100 items (s) | all | 0.6258 | 17.4670 | | | lower | |

PoC v2 không còn lỗi trên 130 ảnh (TP 52, FP 0, FN 0); PoC v1 (14/11, trước khi sửa blur/erase) đạt F1 0.7619 và bỏ lọt 20/52 positive (10 blur severity 2, 10 erase). Chi phí xử lý tăng theo so sánh ảnh cha và bbox; runtime có trong run_info, không phải phép đo detector downstream. Provenance completeness: baseline n/a vì theo đề baseline không ghi provenance; PoC đo trên manifest do chính pipeline xuất (xem bảng).

### Ablation

Tắt lần lượt từng phần của quality gate trên cùng 130 ảnh, không tune lại ngưỡng (`python -m tools.ablation_t03`, chi tiết [ablation_t03.md](ablation_t03.md)):

| Biến thể | F1 v2 | ΔF1 v2 | F1 v1 (14/11) |
|---|---:|---:|---:|
| full | 1.0000 | — | 0.7619 |
| tắt kiểm tra erase | 0.8936 | −0.1064 | 0.7619 |
| tắt kiểm tra blur | 0.8936 | −0.1064 | 0.7619 |
| tắt kiểm tra night | 0.8936 | −0.1064 | 0.5946 |
| tắt kiểm tra glare trong bbox | 0.9804 | −0.0196 | 0.7317 |

Ở v2 mỗi kiểm tra đều có đóng góp riêng; glare-trong-bbox đóng góp ít vì heldout chỉ có 2 ảnh glare_on_object. Ở v1, kiểm tra erase và blur đóng góp 0 vì trên ảnh 1280×720 chúng không bao giờ kích hoạt — đó chính là 2 lỗi được sửa ở v2 (mục 06).

## 05 · Protocol và bất định

Join GT/prediction chính xác theo image_id, positive = unusable theo rule đã chốt. F1 = 2TP/(2TP+FP+FN), precision = TP/(TP+FP), recall = TP/(TP+FN); evaluator giữ giá trị không xác định là null. Không đổi pool/ngưỡng sau khi xem kết quả. 20 ảnh dev của nguồn không tham gia benchmark này.

Đã bootstrap 2.000 lần theo 10 cụm parent_image, seed 42: F1 CI95 baseline [0.520833, 0.591837], PoC v2 [1.000000, 1.000000] (v1: [0.750000, 0.772727]). CI và transform/severity slices do tools.summarize_t03_benchmark sinh ở [chi tiết](t03_benchmark_details.md). Chỉ có 10 parent, phép sinh cùng family với thuật toán, và lựa chọn dữ liệu của nhóm; CI không bao phủ bias do chọn mẫu hoặc sim-to-real.

Ngưỡng v2 chỉ được chọn trên team_dev (20 ảnh dev, F1 dev = 1.0) và được ghi vào config trước khi chạy heldout. Tuy vậy heldout không hoàn toàn "sạch": kết quả v1 trên chính 130 ảnh này (14/11) đã cho biết loại lỗi cần sửa (blur, erase). F1 = 1.0 cũng phản ánh việc gold được sinh bằng đúng các phép biến đổi mà quality gate so khớp với ảnh cha (analysis-by-synthesis); con số này không chứng minh độ chính xác với thời tiết, lóa hay rung máy thật.

## 06 · Hai failure mode bắt buộc của T03

Đề yêu cầu phân tích: (1) synthetic làm mất đối tượng nhưng vẫn giữ label; (2) thiếu trace transform. Cả hai đã xảy ra thật trong repo; metadata và đặc trưng số trong t03_benchmark_details.json (bản v1 trong lịch sử git, commit 56176d4), ảnh before/after giữ local trong out/member6/gold/team_holdout.

1. **Mất đối tượng nhưng giữ label — g0012 (erase)**, parent orig_05, nguồn data/image/6.jpg. GT=1, PoC v1=0 (usable_clean). Bbox sau inpainting còn std=34.74 và color_std=35.38, lớn hơn ngưỡng v1 3.5/6.0: detector v1 giả định vùng xoá gần đồng màu, còn inpainting giữ texture → cả 10 ảnh erase của heldout bị lọt. *Root cause:* đặc trưng dựa trên kiểu fill, không dựa trên việc vùng bbox đã khác ảnh cha. *Sửa (v2):* đo thay đổi trong bbox so với phần còn lại; trên team_dev mọi erase có mean |diff| trong bbox ≥ 14.6 và ≥ 118 lần phần ngoài, ảnh severity 0 tối đa 4.4 và 6 lần. Sau sửa: 10/10 erase bị bắt, Missed label-invalidating edits = 0.
2. **Thiếu trace transform.** Tại commit debf3cf, bước sinh synthetic của pipeline (`src/t03/generate.py`) còn TODO → `transform_manifest.csv` rỗng, Provenance completeness = n/a, quyết định HOLD / MISSING_PROVENANCE_MANIFEST. Trước đó commit f6352ec còn đưa vào một manifest 1 dòng tự ghi "100%" (đã gỡ ở a2d8fbb) và bộ sinh của tools/generate_synthetic_dev.py để glare ở vị trí ngẫu nhiên mà không ghi lại, nên không kiểm được rule "tâm glare trong bbox". *Root cause:* bộ sinh nằm ngoài pipeline và không ghi tham số quyết định nhãn. *Sửa:* bộ sinh vào bước 02; mỗi ảnh có image_id, parent_image, transform, severity, seed, params (tâm glare, obj bị xoá), sha256 cha và con; `provenance.py` dừng pipeline nếu thiếu trường bắt buộc; `decide` trả HOLD nếu manifest rỗng; test `test_t03_default_config_runs_on_team_images` kiểm completeness. Sau sửa: 260/260 dòng đầy đủ, completeness = 1.0, chạy lại cùng seed cho cùng sha256.

Lỗi thứ ba quan sát được (không thuộc danh sách bắt buộc): **g0011 — blur severity 2** bị hạ thành severity 1 vì ngưỡng Laplacian của v1 (0.0032) chọn cho ảnh 320×180; ở 1280×720 tỉ lệ này là 0.0084. Sửa ở v2 bằng ước lượng bán kính blur (mục 03); 10/10 blur severity 2 được bắt.

**Sim-to-real gap:** benchmark đã dùng ảnh gốc thật nhưng weather/corruption vẫn synthetic. Chưa có GT thời tiết tự nhiên, nhãn downstream hay detector để đo gap định lượng. Exposure, lens flare nhiều đỉnh, sensor noise, chuyển động và temporal correlation của camera thật chưa được mô phỏng đầy đủ.

## 07 · Human review và quyết định

Bước rank chọn 10 ảnh (budget B=10) có QC score gần ranh giới usable/unusable nhất (`review_boundary` = 0.63: QC cho usable ≤ 0.48, unusable ≥ 0.78), tức nơi quality gate dễ sai nhất, thay vì 10 ảnh chắc chắn hỏng nhất; mỗi thẻ có `suggested_action` confirm_usable / confirm_unusable. Contact sheet offline hiển thị before/after kèm bbox, bắt buộc tên reviewer; override được audit (`audit_log.jsonl`). Quy trình duyệt: `docs/REVIEW_GUIDE_T03.md`. **Lượt review thật của Member 5 trên run hiện tại: chưa thực hiện** (n_reviewed = 0); file `reports/review_queue_reviewed.csv` hiện có là lượt thử trên bản placeholder cũ (14/11), sẽ được thay.

Chính sách (Member 5, `src/t03/review_flow.py`, ngưỡng trong `configs/lab.yaml`): HOLD nếu thiếu/rỗng transform manifest; REWORK nếu tỷ lệ unusable dự đoán > `max_unusable_rate` = 0.30; còn lại GO. Mỗi quyết định có reason_code + owner và được ghi vào decision_log.

Chạy mặc định trên 30 ảnh nhóm (seed 42): n_generated = 260, provenance đầy đủ; quality gate v2 dự đoán 25/45 ảnh holdout là unusable (0.5556 > 0.30) → **REWORK / UNUSABLE_RATE_EXCEEDED**: lô biến thể hiện tại có quá nhiều ảnh hỏng, cần sinh lại với mức biến đổi nhẹ hơn trước khi đưa vào train. Quyết định này chưa tính reviewer override; cần cập nhật sau lượt review thật của Member 5.

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

Pipeline mặc định (30 ảnh nhóm): `python -m src.main --config configs/lab.yaml --output out/poc --run-name poc --offline --seed 42` → exit 0; lần đầu tự tạo `data/t03_real` từ `data/image` + `data/obj_train_data.zip`. Ablation: `python -m tools.ablation_t03`.

Trên checkout sạch: `python -m pytest -q` → 55 passed, 39 skipped (skip = test cần GT ẩn của giảng viên), gồm 4 test của M6 (`tests/test_t03.py`) và 4 test wiring (`tests/test_t03_pipeline.py`: E2E + tái lập cùng seed, override được audit, thiếu reviewer → exit 2, chạy mặc định trên 30 ảnh với provenance đầy đủ). Lỗi 17 test trước đây chỉ xảy ra trên working tree Windows bị đổi xuống dòng (xem mục 02).

## 09 · Giới hạn và cải tiến

- Ảnh thật lấy từ BDD100K (chỉ nghiên cứu/học tập): không release thương mại, giữ repo private; một người ngoài nhóm nên duyệt lại 10 bbox bị loại trong label_fixes.csv.
- Đánh giá trên nhãn usability thủ công độc lập để kiểm chứng nhãn suy từ operation; không xem bbox zip là nhãn thời tiết tự nhiên.
- Đã xong: generator + manifest trong pipeline (bước 02); quality gate v2 sửa blur/erase trên ảnh 1280×720. Tiếp theo: kiểm quality gate trên biến đổi ngoài phân phối (motion blur, lens flare nhiều đỉnh, sương mù thật) để đo sim-to-real gap.
- Bổ sung human review có người chịu trách nhiệm và audit; hiện mới có queue, chưa có xác nhận của reviewer cho benchmark.
- Không suy từ các số này thành đảm bảo chất lượng production hoặc tuân thủ pháp luật.

## 10 · Đóng góp và tài liệu

Bảng [team_contributions.csv](team_contributions.csv) ghi đóng góp Member 6. Các thành viên khác chưa có ánh xạ tên thật/email đầy đủ nên không tự gán tác giả.

Tài liệu dùng: docs/tracks/T03.md, fixtures/t03_synthetic/README_usability_rule.md, src/metrics/tracks.py::eval_t03, tools/import_team_labels.py và tools/make_gold_set.py. Các metrics/table/details trong reports được sinh bằng script, không nhập số thay evaluator.
