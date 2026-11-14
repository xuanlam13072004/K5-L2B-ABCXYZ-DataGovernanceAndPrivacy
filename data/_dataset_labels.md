# Định nghĩa Nhãn Dữ liệu (Dataset Labels) - Track T03

Tài liệu này mô tả chi tiết danh sách các lớp đối tượng (classes/labels) được trích xuất từ cấu hình huấn luyện (`obj.data` & `obj.names`) phục vụ cho quá trình kiểm soát chất lượng dữ liệu, lập chỉ mục `objects.csv` và kiểm tra Bounding Box (bbox).

## 1. Bảng ánh xạ Class ID và Tên nhãn

Dữ liệu tuân theo định dạng chuẩn của YOLO/Darknet, trong đó chỉ mục (`Class ID`) bắt đầu từ `0`:

| Class ID | Tên nhãn (Label Name) | Mô tả tiếng Việt | Ghi chú kiểm tra |
| :---: | :--- | :--- | :--- |
| **0** | `bus` | Xe buýt | Phương tiện giao thông công cộng cỡ lớn |
| **1** | `fire hydrant` | Trụ cứu hỏa | Xuất hiện tần suất cao trong tập `obj_train_data` |
| **2** | `person` | Người | Đối tượng con người |
| **3** | `stop sign` | Biển báo dừng | Biển báo giao thông hình bát giác |
| **4** | `tennis racket` | Vợt tennis | Dụng cụ thể thao |
| **5** | `traffic light` | Đèn giao thông | Đèn tín hiệu điều khiển giao thông |
| **6** | `train` | Tàu hỏa | Phương tiện đường sắt |
| **7** | `truck` | Xe tải | Phương tiện vận tải hàng hóa |
| **8** | `potted plant` | Cây cảnh trong chậu | Cây trồng đặt trong chậu |
| **9** | *Unassigned / Empty* | *Chưa định nghĩa* | Lưu ý: File `obj.data` khai báo `classes = 10` nhưng `obj.names` chỉ liệt kê 9 tên (từ 0 đến 8). Cần rà soát lại các annotation mang ID `9` (ví dụ trong file `29.txt`). |

---

## 2. Cấu trúc dữ liệu Annotation (YOLO Format)

Mỗi file `.txt` trong thư mục `obj_train_data/` tương ứng với một hình ảnh trong `train.txt` (tổng cộng 30 ảnh từ `001.jpg` đến `30.jpg`). Mỗi dòng trong file annotation đại diện cho một đối tượng với cấu trúc:

```text
<class_id> <x_center> <y_center> <width> <height>
```

* **`class_id`**: Số nguyên đại diện cho loại đối tượng (tham chiếu bảng ở Mục 1).
* **`x_center`, `y_center`**: Tọa độ tâm của Bounding Box, đã được chuẩn hóa (normalize) về khoảng $[0, 1]$ theo chiều rộng và chiều cao của ảnh.
* **`width`, `height`**: Chiều rộng và chiều cao của Bounding Box, đã được chuẩn hóa về khoảng $[0, 1]$.

---

## 3. Lưu ý về Data Governance & Quality Assurance (Member 2)

1. **Cảnh báo lệch số lượng lớp (`classes = 10` vs 9 tên):** Trong file `obj.data`, số lượng lớp được khai báo là `10` (tương ứng ID từ `0` đến `9`), nhưng danh sách `obj.names` hiện chỉ có 9 nhãn. Trong dữ liệu thực tế có xuất hiện đối tượng mang ID `9` (cụ thể ở file `obj_train_data/29.txt`: `9 0.710297 0.686528 0.066609 0.093111`). Cần xác nhận lại với người gán nhãn để bổ sung tên lớp thứ 10 vào `obj.names` nhằm tránh lỗi khi lập chỉ mục `objects.csv`.
2. **Tính toàn vẹn dữ liệu:** Đảm bảo toàn bộ 30 file `.txt` và file `obj.names` không bị thay đổi định dạng dòng (`CRLF` trên Windows) làm sai lệch mã băm SHA256 trong `manifest.csv`.