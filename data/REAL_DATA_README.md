# Mô tả bộ dữ liệu CSV quay thật (test model HCI Slide Gesture)

Bộ dữ liệu được trích xuất từ **67 video quay thật (`1280x720 @ 30.0 fps`)** của **2 người dùng** thông qua [src/09_extract_real_videos.py](../src/09_extract_real_videos.py).

## 1. Quy chuẩn tiền xử lý & Feature (giống 100% schema Jester)

- **Xử lý hướng video (`unmirror`):** Toàn bộ video gốc quay bằng app camera đều bị lật gương sẵn (ngược trái $\leftrightarrow$ phải), nên khi đọc từng khung hình code đã gọi **`cv2.flip(frame, 1)`** để trả về đúng hệ tọa độ không lật của tập train Jester (`next` = vuốt từ trái sang phải của người dùng, tương ứng `dx < 0` trên ảnh đã chuẩn hóa).
- **Tracking & tốc độ lấy mẫu (`7.5 fps`):** Dùng `landmark_utils.create_hands(static_image_mode=False, max_num_hands=1)` chạy MediaPipe **liên tục trên từng frame `30 fps`** để tránh rớt tracking ở pha vuốt nhanh, sau đó lấy mẫu **`1/4` frame** (`--frame-step 4` $\rightarrow$ `7.5 fps`, khớp với [src/05_realtime_gru.py](../src/05_realtime_gru.py)).
- **`70` cột feature đầu (giống hệt Jester frame-level):**
  - `x0..z20` (`63` cột): tọa độ 21 điểm bàn tay (`0.0` nếu không phát hiện được bàn tay).
  - `dx_wrist, dy_wrist, dz_wrist, dx_index, dy_index, dz_index, speed_wrist` (`7` cột): tính theo đúng `sequence_utils.add_motion_features()` (tính độ dời giữa 2 bước lấy mẫu liên tiếp và gán về `0.0` nếu frame trước hoặc frame hiện tại mất tay).

---

## 2. Chi tiết từng file CSV

### (1) `data/real_short_landmarks.csv` — `61` video ngắn `near_miss` (`~1.22 MB` | `841` dòng | `82` cột)
- **Nguồn video:** Toàn bộ **`61` clip ngắn** (`1.7s–5.1s`) mô phỏng các tình huống dễ gây nhận nhầm (*Midas touch*), toàn bộ mang `label = none`, `scenario = near_miss`, `split = test`.
- **Phân bố chi tiết `61` clip theo khoảng cách & kịch bản (`sub_category`):**
  - **Nhóm `Gần` (`37` clip):**
    - `basic`: `4` clip (`2` near next, `2` near prev)
    - `lấy đồ vật`: `10` clip (`5` near next, `5` near prev)
    - `mời học sinh phát biểu`: `12` clip (`5` near next, `7` near prev)
    - `khi thực hiện prev/next nhưng khi thu tay về trông giống lệnh ngược`: `11` clip (`5` near next, `6` near prev)
  - **Nhóm `Xa` (`24` clip):**
    - `basic`: `4` clip (`2` near next, `2` near prev)
    - `mời học sinh phát biểu`: `10` clip (`5` near next, `5` near prev)
    - `khi thực hiện prev/next nhưng khi thu tay về trông giống lệnh ngược`: `10` clip (`5` near next, `5` near prev)
- **Cách đóng gói trong code:** Giống `src/08_jester.py`, mỗi clip cắt giữ đoạn `[frame thấy tay đầu tiên .. frame thấy tay cuối cùng]` (có **`2` clip** không phát hiện được bàn tay nào nên được giữ dưới dạng 1 dòng toàn `0` với `frame_idx = -1`).
- **Cột metadata thêm:** `scenario`, `sec`, `distance` (`Gần`/`Xa`), `near_group`, `sub_category`, `source_file`.

---

### (2) `data/real_long_landmarks.csv` — `6` video thuyết trình dài liên tục (`~11.80 MB` | `8,349` dòng | `82` cột)
- **Nguồn video:** Chuỗi frame liên tục từ đầu tới cuối của **`6` bài thuyết trình thật** (~`18.5` phút tổng thời lượng), giữ đủ cả các bước không có tay để test mô hình theo luồng stream thực tế ở `7.5 fps`:
  - **`nguoi_1_2026_11_21_20_43_45`** (`337.7s`, `2,534` bước): `9` lệnh `next`, `17` đoạn `near_miss`, `27` đoạn `background`.
  - **`nguoi_2_video_1`** (`253.7s`, `1,904` bước): `6` lệnh `next`.
  - **`nguoi_2_video_2`** (`176.9s`, `1,328` bước): `3` lệnh `previous`.
  - **`nguoi_2_video_3`** (`132.1s`, `992` bước): `2` lệnh `next`, `1` lệnh `previous`.
  - **`nguoi_2_video_4`** (`117.5s`, `882` bước): `4` lệnh `next`, `2` đoạn `near_miss`.
  - **`nguoi_2_video_5`** (`94.4s`, `709` bước): `3` lệnh `next`, `1` lệnh `previous`, `2` đoạn `near_miss` (đã gồm đoạn `0:47–0:53`).
- **Gán nhãn theo dòng (`label`, `scenario`, `trial_id`):** Mỗi khoảng sự kiện `intentional`/`near_miss` được kéo dài thêm tối đa `+1.5s` ở đuôi (không đè lên sự kiện kế tiếp) để bao phủ độ trễ cửa sổ trượt của GRU; các khoảng còn lại mang nhãn `scenario = background`, `label = none`.
- **Cột metadata thêm:** `scenario`, `sec`, `trial_id`, `speaker`, `source_file`, `step_idx`.

---

### (3) `data/real_long_annotations.csv` — Bảng mốc nhãn gốc của `6` video dài (`~12.6 KB` | `77` dòng | `9` cột)
- Hợp nhất từ file gán nhãn thủ công của Người 1 (`slide_gesture_annotations-v5.csv`: `53` đoạn chi tiết gồm `NEXT` $\rightarrow$ `intentional/next`, `AMBIGUOUS` $\rightarrow$ `near_miss/none`, `OTHER` $\rightarrow$ `background/none`) và Người 2 (`Gán nhãn Video.txt` kèm mốc `0:47–0:53: near_miss` ở Video 5: `24` mốc sự kiện).
- Cột: `video_id, speaker, source_file, start_sec, end_sec, trial_type, intended_action, gesture_type, description`. Dùng để đối chiếu timestamp gốc khi debug/tự cắt cửa sổ trên `real_long_landmarks.csv`.

---

### (4) `data/real_clips_landmarks.csv` — Tập `164` clip gộp chuẩn pipeline repo (`~4.91 MB` | `3,562` dòng | `80` cột)
- **Mục đích:** Gộp toàn bộ mẫu về dạng từng clip ngắn độc lập giống chuẩn Jester để nạp thẳng vào `jester_data.load_videos(..., 16)` và `src/06_simulate_stream.py`.
- **Tổng `164` clip gồm:**
  - `61` clip `short_near_miss` (từ `real_short_landmarks.csv`)
  - `29` clip `long_intentional`: **`24` `next` + `5` `previous`** (cắt từ `6` video dài, trim `[first_detected .. last_detected]`)
  - `21` clip `long_near_miss`: `17` của Người 1 + `4` của Người 2 (trim `[first_detected .. last_detected]`)
  - `53` clip `long_background`: cắt từ các khoảng nói chuyện bình thường giữa các lệnh (giới hạn tối đa `32` bước $\approx 4.3s$/clip).

---

### (5) `data/real_predictions.csv` — Prediction log sẵn từ `models/gru.keras` (`~1.79 MB` | `9,767` dòng | `18` cột)
- **Không dùng để test model mới**, đây là kết quả đã chạy sẵn của `models/gru.keras` liên tục trên `61` clip ngắn + `6` video dài (`164` `trial_id`), dùng làm đầu vào cho `src/evaluate_policies.py`.

---

## 3. Cách đưa vào code để chạy thử model của bạn

### Lưu ý trước khi chạy
Cần dùng bản `src/jester_data.py` đã cập nhật (hỗ trợ đọc trực tiếp cột `scenario`/`label` của các file `real_*.csv` thay vì chỉ lọc tên raw label của Jester).

### Cách A — Test nhanh bằng script sẵn của repo (`real_clips_landmarks.csv` hoặc `real_short_landmarks.csv`)

```powershell
python src/06_simulate_stream.py --data data/real_clips_landmarks.csv --model models/gru.keras --output data/real_clips_predictions.csv
python src/evaluate_policies.py --input data/real_clips_predictions.csv --output-dir reports/real_clips_eval --fps 7.5 --hold-frames 3 --cooldown-frames 8 --margin-threshold 0.20
```

### Cách B — Đọc bằng Python để test model hoặc kiến trúc tự viết

**1. Dạng clip chuẩn hóa `16` bước (dùng cho `real_clips_landmarks.csv` hoặc `real_short_landmarks.csv`):**

```python
import sys
sys.path.insert(0, "src")

from jester_data import aligned_inputs, load_videos

ds = load_videos("data/real_clips_landmarks.csv", length=16)
X = aligned_inputs(ds.tracks)   # shape: (164, 16, 70)
y_true = ds.meta["target"]      # next / previous / zoom_in / zoom_out / none
scenario = ds.meta["scenario"]  # intentional / background / near_miss
```

**2. Dạng luồng liên tục theo từng frame trên `6` video dài (`real_long_landmarks.csv`):**

```python
import sys
sys.path.insert(0, "src")

import pandas as pd
from jester_data import LANDMARK_COLUMNS, MOTION_COLUMNS

df = pd.read_csv("data/real_long_landmarks.csv")
feature_cols = LANDMARK_COLUMNS + MOTION_COLUMNS  # 70 cột theo đúng thứ tự
for vid, vdf in df.groupby("video_id", sort=False):
    X_seq = vdf[feature_cols].to_numpy(dtype="float32")  # chuỗi liên tục ~7.5 fps
    # Trượt cửa sổ 16 bước (hoặc theo độ dài model của bạn) trên X_seq
```
