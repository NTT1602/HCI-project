# Hand Gesture Slide Controller

Project HCI điều khiển slide bằng cử chỉ tay: video → MediaPipe hand landmarks → GRU sequence classifier → gate chống trigger lặp.

## Cài đặt

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn activate, chạy trực tiếp `.venv\Scripts\python.exe` hoặc đổi execution policy theo chính sách máy của bạn.

## Pipeline nghiên cứu Midas touch (chạy 1 lệnh)

```powershell
python src/run_pipeline.py              # train GRU -> mô phỏng luồng -> so sánh C1-C4 + quét ngưỡng
python src/run_pipeline.py --quick      # chạy thử nhanh (~4 phút)
python src/run_pipeline.py --skip-train # dùng lại models/gru.keras
```

| Bước | Script | Đầu ra |
|---|---|---|
| 1. Train GRU 5 lớp (`next`, `previous`, `zoom_in`, `zoom_out`, `none`) theo cột `split` của CSV | `src/04_train_gru.py` | `models/gru.keras` (+ `.labels.json`, `.meta.json`), `reports/classifier_report.txt`, `reports/confusion_matrix.csv` |
| 2. Mô phỏng luồng camera trên tập test: không tay → cử chỉ → giữ tay yên, log prediction từng frame | `src/06_simulate_stream.py` | `data/predictions.csv` |
| 3. Chạy cùng log qua C1 fixed, C2 adaptive, C3 fixed+abstention, C4 adaptive+abstention | `src/evaluate_policies.py` | `reports/evaluation/policy_summary.csv`, `threshold_sweep.csv`, `trial_results.csv`, `policy_events.csv` |

Kịch bản (`src/jester_data.py`):

- **intentional**: Swiping Left/Right, Zooming In/Out → lệnh.
- **background**: Doing other things, No gesture → `none` (có trong train).
- **near_miss**: Thumb Up, Thumb Down, Stop Sign → `none`, **không đưa vào train** (mặc định) để kiểm tra cử chỉ tay có chủ ý mà model chưa từng thấy. Thêm `--train-near-miss` nếu muốn đưa vào.

CSV hiện tại là 1 dòng/video (pose frame giữa + vector đầu→cuối), nên `jester_data.py` dựng quỹ đạo 16 bước bằng cách tịnh tiến pose theo vector chuyển động (ngón trỏ có thêm độ dời riêng). Đây là xấp xỉ; nếu có CSV nhiều dòng/video từ `08_jester.py` (có `frame_idx`) thì pipeline tự dùng quỹ đạo thật.

Đơn vị thời gian là bước chuỗi (~8 fps: 16 bước ≈ 2 s). Tham số policy mặc định: hold 3 bước, cooldown 8 bước. Realtime lấy 1/4 frame webcam để khớp tốc độ này.

Chỉ số: `command_success_rate`, `command_miss_rate`, `wrong_command_rate`, `false_triggers_per_minute`, `background_trigger_rate`, `near_miss_trigger_rate` (tỉ lệ trial có ≥1 lần kích hoạt nhầm), `median_latency_ms` (tính từ lúc bắt đầu cử chỉ), `abstention_rate`.

Thu thử nghiệm thật bằng webcam theo cùng format rồi đánh giá:

```powershell
python src/05_realtime_gru.py --log data/real_trials.csv --trial-id p01_t01 --trial-type background --intended none
python src/evaluate_policies.py --input data/real_trials.csv --fps 7.5 --hold-frames 3 --cooldown-frames 8 --margin-threshold 0.2
```

## Jester + GRU

Jester phù hợp hơn cho gesture động. Tải `Jester_zip1`, `Jester_zip2`, `Jester_zip3` và file `Labels`; ba file zip đều chứa các phần video khác nhau, không tách theo nhãn. Sau khi giải nén, lọc bốn nhãn cần dùng và tạo chuỗi landmarks:

```powershell
python src/04_train_gru.py --data data/jester_landmarks.csv --model models/gru.keras
python src/05_realtime_gru.py --model models/gru.keras
```

Script đọc trực tiếp CSV landmark đã tạo, nhóm frame theo `video_id`, rồi resample mỗi video thành 16 bước. Mapping nhãn: `Swiping Right` → `next`, `Swiping Left` → `previous`, `Zooming In/Out With Two Fingers` hoặc `With Full Hand` → `zoom_in/zoom_out`, lớp nền → `none`. GRU dùng 70 features mỗi frame (63 landmarks + 7 motion features). Nếu CSV chỉ có một dòng/video, quỹ đạo được dựng xấp xỉ từ vector chuyển động (xem mục pipeline ở trên).

Để tạo CSV nhiều dòng/video từ các thư mục JPG Jester, chạy lại extractor với output mới (CSV cũ là format một dòng/video):

```powershell
python src/08_jester.py --manifest <manifest.csv> --output data/jester_landmarks_frames.csv --step 3
python src/04_train_gru.py --data data/jester_landmarks_frames.csv --model models/gru.keras
```

TensorFlow được ghim ở `2.16.2` để tương thích với MediaPipe trên Windows/Python 3.12.

## 20BN-JESTER

Tổ chức bộ dữ liệu Jester cho bài toán điều khiển slide (lọc 11 nhãn liên quan: `Swiping Left`, `Swiping Right`, `Stop Sign`, `No gesture`, `Doing other things`, `Thumb Up`, `Thumb Down`, `Zooming In/Out`):

```powershell
python src/organize_jester.py
```

Dữ liệu được tổ chức theo cấu trúc chuẩn `train/`, `validation/`, `test/` tại `D:\Data HCI\slide_gestures` kèm file chỉ mục `manifest.csv`.

### Trích xuất Landmarks + Vector chuyển động (Option 2)

Trích xuất 63 đặc trưng MediaPipe hand landmarks tại keyframe kết hợp 7 đặc trưng vector chuyển động động học (`dx_wrist`, `dy_wrist`, `dz_wrist`, `dx_index`, `dy_index`, `dz_index`, `speed_wrist`) để phân biệt chính xác cử chỉ vuốt:

```powershell
# Chạy thử nghiệm nhanh (ví dụ 100 mẫu mỗi nhãn):
python src/08_jester.py --limit-per-class 100 --output data/jester_landmarks.csv

# Hoặc trích xuất toàn bộ:
python src/08_jester.py --output data/jester_landmarks.csv
```

## Kiểm thử

```powershell
python -m pytest -q
python -m compileall src tests
```



## Lưu ý

- `models/` và dữ liệu thu thập bị ignore khỏi Git để không đẩy file lớn/dữ liệu camera.
- `src/gate.py` chuyển prediction ổn định thành `NEXT SLIDE`, `PREVIOUS SLIDE`, `ZOOM IN`, `ZOOM OUT`; có thể nối thêm `pyautogui` hoặc PowerPoint COM sau khi model ổn định.
- Webcam và MediaPipe cần môi trường desktop có quyền camera; không thể xác nhận hình ảnh realtime trong terminal tự động.
