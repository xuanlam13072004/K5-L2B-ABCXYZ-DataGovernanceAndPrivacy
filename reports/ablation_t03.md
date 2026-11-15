# T03 ablation — sinh bằng `python -m tools.ablation_t03`

Mỗi dòng tắt **một** phần của bộ lọc QC (chỉ đổi `params.thresholds` trong bản sao tạm của `configs/lab.yaml`), chạy lại pipeline và chấm bằng `src.evaluate`. Không tune ngưỡng ở đây.

| Biến thể | Gold set | N | F1 | ΔF1 so với full | Precision | Recall | Erase bị lọt |
|---|---|---:|---:|---:|---:|---:|---:|
| full | team_holdout | 130 | 1.0000 | — | 1.0000 | 1.0000 | 0 |
| no_erase_check | team_holdout | 130 | 0.8936 | -0.1064 | 1.0000 | 0.8077 | 10 |
| no_glare_box_check | team_holdout | 130 | 0.9804 | -0.0196 | 1.0000 | 0.9615 | 0 |
| no_blur_check | team_holdout | 130 | 0.8936 | -0.1064 | 1.0000 | 0.8077 | 0 |
| no_night_check | team_holdout | 130 | 0.8936 | -0.1064 | 1.0000 | 0.8077 | 0 |

## F1 theo kiểu biến đổi

| Biến thể | Gold set | blur | erase | fog | glare | night |
|---|---|---:|---:|---:|---:|---:|
| full | team_holdout | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| no_erase_check | team_holdout | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| no_glare_box_check | team_holdout | 1.0000 | 1.0000 | 1.0000 | 0.9091 | 1.0000 |
| no_blur_check | team_holdout | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| no_night_check | team_holdout | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

Gold set: nhãn suy ra từ phép biến đổi theo `README_usability_rule.md` (`tools/make_gold_set.py`), không phải nhãn ẩn của giảng viên.
