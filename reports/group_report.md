# Báo cáo nhóm — Day 06 Data Lab · T03 Synthetic-vs-Real

> Bản báo cáo dựa trên fixture, config, source và log đang có trong checkout ngày 2026-11-14. Không điền điểm benchmark khi không có GT/eval output để tái lập.

## 00 · Thông tin nhóm

| Mục | Giá trị |
|---|---|
| Group ID | G01 (`configs/lab.yaml`) |
| Challenge | T03 · Synthetic-vs-Real: tạo và kiểm dữ liệu khó |
| Thành viên / vai trò | M1 Tech Lead & Pipeline Integrator · M2 Data Governance & Provenance · M3 Synthetic Generator & Baseline · M4 Quality Gate Algorithm · M5 Human-in-the-Loop & Decision · M6 QA, Benchmarking & Reporting |
| Tên thành viên | Chưa có trong tài liệu được cung cấp; đang dùng mã M1–M6 |
| Git URL | Chưa ghi trong report; checkout có remote `origin` |
| Tag `demo-180m` / `final-24h` | Chưa có tag trong checkout hiện tại |
| Deadline | Mốc +24h theo role assignment; thời điểm cụ thể chưa được cung cấp |
| Ngày lập | 2026-11-14 |

## 01 · Bài toán và tiêu chí quyết định

- **Ai chịu ảnh hưởng:** nhóm gán nhãn và nhóm huấn luyện mô hình thị giác xe tự hành; lỗi QC có thể đưa ảnh không dùng được hoặc nhãn sai vào tập train.
- **Điểm nghẽn:** biến đổi synthetic có thể làm mất khả năng quan sát đối tượng hoặc làm nhãn không còn đúng, trong khi ảnh vẫn trông hợp lệ ở chỉ số toàn khung hình.
- **Đầu vào → đầu ra:** ảnh gốc, bảng bbox và ảnh holdout đã biến đổi → `qc.csv` gồm `image_id`, `pred_unusable` cùng score/reason; provenance manifest, review queue, audit log và quyết định.
- **GO / HOLD / REWORK:** M5 phát hành GO nếu provenance có dữ liệu và tỷ lệ unusable dự đoán không vượt `max_unusable_rate=0.30`; HOLD nếu thiếu provenance; REWORK nếu tỷ lệ vượt ngưỡng. Đây là chính sách của PoC, không phải phê duyệt phát hành dữ liệu thật.
- **Giả định:** fixture là synthetic, không chứa PII thật; ngoài fixture cần domain owner xác nhận usability, DPO/legal xác nhận các nghĩa vụ áp dụng. Không tuyên bố tuân thủ luật từ QC tự động.

## 02 · Dữ liệu và provenance

| Split / bảng | N | Nguồn / license | Synthetic? |
|---|---:|---|---|
| Originals | 30: 20 dev + 10 holdout | `fixtures/t03_synthetic/manifest.csv`; CC0-1.0, được tạo cho AI20K Day06 | Có |
| Holdout transformed | 45 ảnh, 5 biến thể cho mỗi 10 ảnh gốc holdout | Cùng fixture và license trong manifest | Có |
| Objects | Bbox theo ảnh gốc | `fixtures/t03_synthetic/objects.csv` | Có |
| Ảnh thực | 0 trong checkout được đánh giá | `data/real/` không có trong đầu vào được cung cấp | Không áp dụng |

Schema ảnh: `image_id`, `path`, `split`; bbox: `image_id`, `obj_id`, `class`, `x`, `y`, `w`, `h`. Seed chuẩn là 42. Split theo ID ảnh gốc (dev/holdout), không chia ngẫu nhiên từng biến thể. Đã chạy `python -m tools.verify_manifests`: cả 14/14 manifest fixture đang fail hash; riêng T03 có 5 hash lệch. Blob trong Git khớp hash manifest, còn byte trong working tree khác; `core.autocrlf=true` và CRLF hiện diện, nên nguyên nhân nhiều khả năng là xử lý line ending trên Windows. Đây là lỗi cần M1/M2 xử lý ở `.gitattributes`/checkout trước khi bật strict provenance.

**Giới hạn provenance:** `src/t03/provenance.py` cung cấp schema manifest, nhưng hiện generator `src/t03/generate.py::generate_dev_variants()` trả DataFrame rỗng (TODO). Vì vậy chưa tạo được provenance cho ảnh sinh từ pipeline ở checkout này; `decision_log.csv` có hai bản ghi HOLD với `MISSING_PROVENANCE_MANIFEST`.

GT nằm trong `fixtures/t03_synthetic/gt/` theo track docs và bị loại khỏi learner repo bởi `.gitignore`. Nội dung GT không được dùng hoặc suy đoán trong báo cáo này. Đánh giá holdout cần instructor overlay.

## 03 · Thiết kế

Luồng dự kiến: `originals + objects` → sinh night/glare/fog/blur/erase từ dev → ghi lineage → so ảnh holdout với ảnh cha và bbox → QC → top-10 review → override/audit → quyết định.

PoC QC (`src/t03/qc.py`) so sánh cường độ sáng, độ tương phản, Laplacian và ảnh cha; dò tâm glare đối với bbox; dò vùng bbox gần đồng màu để nhận diện erase. Ngưỡng được đọc từ `configs/lab.yaml`. Baseline (`src/pipeline_baseline.py`) dùng blur Laplacian và độ tối toàn ảnh, không phân tích bbox hay provenance.

Không có model pretrained, API ngoài hoặc dữ liệu thật trong đường chạy được mô tả. Trade-off chính là rule QC dễ chạy offline và giải thích được, nhưng nhạy với ngưỡng và không học được hình dạng nhiễu camera ngoài đời.

Tài liệu thiết kế dùng trong repo: [Track T03](../docs/tracks/T03.md), [usability rule](../fixtures/t03_synthetic/README_usability_rule.md), evaluator chuẩn `src/metrics/tracks.py::eval_t03` và contract ở [metric_contract.md](metric_contract.md).

## 04 · Baseline vs PoC

| Hạng mục | Baseline | PoC |
|---|---|---|
| Phương pháp | Laplacian + ngưỡng tối, toàn ảnh | So ảnh biến đổi với ảnh cha; kiểm tra bbox, glare/erase và mức độ biến đổi |
| Input | Cùng fixture T03 khi chạy | Cùng fixture T03 khi chạy |
| Seed / heldout | Seed 42 / 45 ảnh holdout theo contract | Seed 42 / cùng pool theo contract |
| F1 / precision / recall | Chưa chấm được | Chưa chấm được |

Không có `fixtures/t03_synthetic/gt/holdout_gt.csv` trong learner checkout và không có `reports/metrics_baseline.json` / `reports/metrics_poc.json`. Vì vậy `src.evaluate` chưa thể sinh metrics JSON; `src.compare` chưa thể tạo bảng kết quả hợp lệ. File `reports/table1_baseline_vs_poc.md` sẽ được sinh bởi script sau khi instructor cung cấp GT overlay và hai run hoàn chỉnh. Không thay thế kết quả bằng số giả hoặc nhãn holdout tự suy đoán.

## 05 · Protocol đánh giá

Primary metric là Transform QC F1, positive = unusable; công thức và join chính xác theo `image_id` được khai báo trong [metric_contract.md](metric_contract.md). Candidate pool là 45 holdout; threshold policy dùng config hiện tại, seed 42. Contract chưa chứng minh các ngưỡng được tune trên dev gold: dev gold cũng chưa được materialize ở checkout này. Không chỉnh threshold sau khi xem holdout.

Khi GT sẵn có, chạy đúng các lệnh trong metric contract cho baseline và PoC, xuất metrics JSON và `table1_baseline_vs_poc.md`. Báo slice theo transform và missed `erase`; bootstrap CI theo cụm ảnh gốc do 45 biến thể phụ thuộc vào 10 ảnh cha.

## 06 · Failure cases và sim-to-real gap

Các mục dưới đây là **failure modes cần kiểm tra theo rule và logic hiện tại**, không phải false negative/false positive đã xác nhận trên GT holdout:

1. **Erase cục bộ, nhãn bbox còn nguyên:** rule coi đây là unusable. Bộ dò chỉ chạy khi thống kê toàn ảnh gần như không đổi; nếu vùng bị xóa còn nhiều texture/viền hoặc thống kê ảnh vượt điều kiện kích hoạt, độ lệch chuẩn trong bbox có thể không xuống ngưỡng và ảnh bị bỏ lọt. Cần kiểm theo `reason=object_removed_label_kept` trên `erase` slice.
2. **Glare severity 1 chạm bbox nhưng không tạo đỉnh mạnh:** rule coi glare severity ≥1 trong bbox là unusable. Gate dùng điều kiện toàn ảnh (`mean_ratio`, `p99_diff`) và ước lượng tâm Gaussian; flare lệch tâm, nhiều đỉnh hoặc bị clipping có thể làm sai tâm/ngưỡng, gây miss hoặc cảnh báo nhầm. Bài test đơn vị bao phủ glare Gaussian lý tưởng nằm trong bbox, chưa thay thế kiểm tra holdout.

**Sim-to-real gap:** fixture mô phỏng biến đổi tham số hóa (night, glare Gaussian, fog, blur, erase). Camera thật có auto-exposure, rolling shutter, chuyển động, lens flare đa hình, noise phụ thuộc cảm biến, mưa/bẩn ống kính và tương quan theo thời gian. Checkout không có ảnh thật hay detector downstream để đo hiệu năng nhận diện; do đó chưa thể định lượng sim-to-real gap. Bước tiếp theo là đánh giá riêng trên real data được cấp phép, có protocol gán nhãn rõ ràng, không đưa ảnh có người/biển số lên repo công khai.

## 07 · Human review và quyết định

M5 tạo queue top-10 theo score, contact sheet HTML offline hiển thị ảnh gốc/biến đổi và bbox, nhận reviewer override rồi ghi `audit_log.jsonl`. Decision yêu cầu `reason_code` và `owner`. Log hiện có ở `reports/decision_log.csv` ghi hai lần HOLD do manifest rỗng/thiếu. Các lần đó cho thấy guard provenance đang chặn release; chúng không phải kết quả benchmark chất lượng ảnh.

## 08 · Tái lập và kiểm thử

Lệnh đường chạy chính theo repo:

```bash
python -m src.main --config configs/lab.yaml --track t03 --offline --seed 42 --run-name poc --output out/poc
python -m src.evaluate --track t03 --pred out/poc/qc.csv --run-name poc --out reports/metrics_poc.json
python -m src.compare reports/metrics_baseline.json reports/metrics_poc.json --out reports/table1_baseline_vs_poc.md
pytest -q
```

Môi trường: Python 3.13.14, Windows 11. `python -m pytest -q tests/test_t03.py` pass 4 tests. Toàn bộ `python -m pytest -q` có 17 failures: 14 kiểm thử hash fixture và 3 kiểm thử E2E T03 bị dừng bởi strict provenance do 5 hash lệch T03. Các pipeline tests còn kiểm reproducibility, review override/audit và từ chối review thiếu reviewer. Tài khoản shell không có lệnh `pytest` trực tiếp; dùng module invocation như trên. Pipeline chưa là clean-start hoàn chỉnh vì generator dev còn TODO; hiện E2E bị chặn ngay từ bước load bởi hash mismatch trước khi tới generator.

## 09 · Giới hạn và đạo đức

- Kết quả chỉ áp dụng cho fixture synthetic nhỏ; không khái quát thành đảm bảo chất lượng ảnh thực.
- Rule threshold có thể gây false alarm hoặc miss; dùng reviewer và audit log trước quyết định release.
- Manifest fixture khai báo CC0-1.0; dữ liệu thật cần kiểm license/quyền sử dụng riêng.
- Không có PII thật trong fixture theo docs; dữ liệu thật, nếu được cấp, không commit lên repo công khai/cloud.
- Không kết luận tuân thủ GDPR hoặc luật Việt Nam; cần DPO/legal xác nhận cho use case thực tế.
- Scale lên 10× cần đo latency và độ ổn định theo thiết bị, scene và nguồn camera; hiện chưa có phép đo.

## 10 · Đóng góp nhóm

| Thành viên | Vai trò / bằng chứng |
|---|---|
| M1 | Integration, CLI, cấu hình; xem `reports/team_contributions.csv` và lịch sử Git |
| M2 | Fixture, hash và provenance |
| M3 | Generator và baseline; generator hiện còn TODO |
| M4 | Quality gate `src/t03/qc.py` |
| M5 | Review flow, contact sheet, decision |
| M6 | Tests T03, metric contract, báo cáo, slide demo; Git author `maawongf <maawongf@gmail.com>`; commit deliverable `e0b7c45` |

Không gán tên thật hoặc email Git cho M1–M5 khi tài liệu chưa ánh xạ danh tính. SHA của commit M6 được ghi trong [team_contributions.csv](team_contributions.csv); các mapping còn thiếu cần nhóm bổ sung.

## 11 · Tài liệu tham chiếu

| Tài liệu | Ý nghĩa |
|---|---|
| `docs/tracks/T03.md` | Schema dữ liệu, output, primary metric, giới hạn eval và mốc bài lab |
| `fixtures/t03_synthetic/README_usability_rule.md` | Định nghĩa positive `unusable` được dùng làm GT |
| `src/metrics/tracks.py::eval_t03` | Thực thi metric, phép nối ID và slice chuẩn; không sửa metric để làm đẹp số |
