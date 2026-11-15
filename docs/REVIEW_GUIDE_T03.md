# Hướng dẫn lượt review thật — Member 5 (T03)

Mục tiêu: một người thật xem 10 ảnh mà quality gate **phân vân nhất**, xác nhận hoặc sửa kết luận
"dùng được / không dùng được", để pipeline áp override, ghi audit log và ra quyết định cuối.
Không được điền thay người khác: tên reviewer là người thật sự đã xem ảnh.

## 1. Chạy pipeline (PowerShell, trong thư mục repo)

```powershell
git pull
python -m src.main --config configs/lab.yaml --output out/poc --run-name poc --offline --seed 42
```

Dòng cuối phải có `status=0`. Lần đầu chạy, pipeline tự tạo `data/t03_real` (khoảng 20–60 giây).

## 2. Mở trang duyệt

Mở file `out/poc/contact_sheet.html` bằng trình duyệt (Chrome/Edge). Trang chạy offline.

Mỗi thẻ có ảnh **before** (ảnh gốc) và **after** (ảnh đã biến đổi), cùng:
- `reason`: quality gate nghĩ ảnh bị gì (vd. `severity_2_blur`, `severity_1_glare_outside_bbox`).
- `suggested_action`: `confirm_unusable` (QC nói KHÔNG dùng được) hoặc `confirm_usable` (QC nói dùng được).

Quy tắc dùng được (đã chốt, `fixtures/t03_synthetic/README_usability_rule.md`) — ảnh **không dùng được** nếu:
1. biến đổi quá nặng (mức 2): quá tối, quá mờ, sương quá dày, lóa quá mạnh;
2. vùng lóa có tâm rơi lên một xe / người / biển báo đã gán nhãn;
3. một vật thể đã bị xoá khỏi ảnh nhưng nhãn của nó vẫn còn.

## 3. Điền quyết định cho từng thẻ

| Trường hợp | Chọn ở ô decision | Ô override value | Ô reason / note |
|---|---|---|---|
| Đồng ý với QC | `approve` | để trống | lý do ngắn, vd. "xe vẫn nhìn rõ" |
| QC nói dùng được nhưng thật ra KHÔNG dùng được | `edit` | `1` | vd. "lóa che mất biển báo" |
| QC nói không dùng được nhưng thật ra DÙNG được | `edit` | `0` | vd. "mờ nhẹ, vẫn đếm được xe" |
| Không chắc, cần người khác xem | `escalate` | để trống | ghi điều đang phân vân |

Điền tên thật vào ô **Reviewer (bắt buộc)** ở đầu trang, rồi bấm **Export decisions CSV**.

## 4. Đưa kết quả vào pipeline

1. Trình duyệt tải về file `review_queue_reviewed.csv` (thường ở thư mục Downloads).
2. Chép file đó vào `out/poc/review_queue_reviewed.csv`.
3. Chạy lại đúng lệnh ở bước 1. Pipeline sẽ áp override, ghi `out/poc/audit_log.jsonl` và quyết định mới.

Nếu thiếu tên reviewer, pipeline dừng với `status=2` — đó là cố ý.

## 5. Lưu bằng chứng vào repo (thư mục out/ không được commit)

```powershell
Copy-Item out/poc/review_queue_reviewed.csv reports/review_queue_reviewed.csv -Force
Copy-Item out/poc/audit_log.jsonl reports/audit_log.jsonl -Force
Copy-Item out/poc/decision.json reports/decision.json -Force
Copy-Item out/poc/decision_log.csv reports/decision_log.csv -Force
git add reports/review_queue_reviewed.csv reports/audit_log.jsonl reports/decision.json reports/decision_log.csv
git commit -m "review(t03): lượt review thật 10 ảnh + audit log + quyết định cuối"
git push
```

Sau đó báo Lead/Member 6 để cập nhật mục 07 của `reports/group_report.md` (số ảnh đồng ý / sửa / escalate,
quyết định cuối).
