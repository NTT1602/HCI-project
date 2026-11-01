# Hand Gesture Slide Controller

Project HCI điều khiển slide bằng cử chỉ tay: webcam → 21 hand landmarks → thu thập CSV → Random Forest → realtime prediction → gate chống trigger lặp.

## Cài đặt

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn activate, chạy trực tiếp `.venv\Scripts\python.exe` hoặc đổi execution policy theo chính sách máy của bạn.

## Chạy theo thứ tự

Từ thư mục gốc repo, chạy từng bước; nhấn `q` để thoát webcam:

```powershell
python src/01_webcam.py
python src/02_landmarks.py
python src/03_collect.py --output data/landmarks.csv --user-id local
python src/04_train.py --data data/landmarks.csv --model models/rf.pkl
python src/05_realtime.py --model models/rf.pkl
python src/run_gate_realtime.py --model models/rf.pkl
```

## Nghiên cứu Midas touch: prediction log và evaluator

Prediction logger ghi kết quả model theo từng frame, nhưng không áp dụng gate. Vì vậy cùng một log có thể được replay qua cả 4 chính sách một cách công bằng.

Mỗi trial là một lần quay riêng. Với lệnh có chủ ý, `--trial-type intentional` và đặt `--intended-action`; với hoạt động nền hoặc near-miss, đặt `--intended-action none`.

Ví dụ quay các trial:

```powershell
python src/record_predictions.py --model models/rf.pkl --output data/predictions.csv --trial-id intentional-next-01 --trial-type intentional --intended-action next
python src/record_predictions.py --model models/rf.pkl --output data/predictions.csv --trial-id background-point-01 --trial-type background --intended-action none
python src/record_predictions.py --model models/rf.pkl --output data/predictions.csv --trial-id near-miss-next-01 --trial-type near_miss --intended-action none
```

Nhấn `q` để kết thúc từng trial. Logger lưu `confidence`, `second_confidence`, `margin`, `predicted_label` và metadata trial.

Replay cùng log qua C1–C4:

```powershell
python src/evaluate_policies.py --input data/predictions.csv --output-dir data/evaluation
```

Kết quả:

- `data/evaluation/policy_summary.csv`: bảng so sánh 4 policy.
- `data/evaluation/trial_results.csv`: kết quả từng trial.
- `data/evaluation/policy_events.csv`: các event mà mỗi policy phát ra.

Các metric chính là `false_triggers_per_minute`, `command_miss_rate`, `mean_latency_ms`, `command_success_rate` và `abstention_rate`.
Ở bước collect, chọn nhãn bằng phím:

- `1`: `next`
- `2`: `previous`
- `3`: `play_pause`
- `0`: `none`

Giữ phím nhãn không cần thiết: sau khi chọn nhãn, mỗi frame có tay sẽ được ghi. Thu tối thiểu vài trăm mẫu mỗi nhãn, đa dạng khoảng cách/góc/người, rồi mới train.

## HaGRID

Tạo manifest CSV có 3 cột `image_path,label,user_id`, sau đó:

```powershell
python src/06_hagrid.py --manifest data/hagrid_manifest.csv --output data/hagrid_landmarks.csv --extract
```

Script chia test theo `user_id` để tránh cùng một người xuất hiện ở cả train và test, sau đó train Random Forest và in accuracy/report. Dataset HaGRID có thể có schema/nhãn khác, vì vậy cần map nhãn trong manifest về `next`, `previous`, `play_pause`, `none` trước khi train so sánh.

## Kiểm thử

```powershell
python -m pytest -q
python -m compileall src tests
```

## Tạo GitHub repository

Repo local đã sẵn sàng. Để tạo repo GitHub và push, cài GitHub CLI rồi đăng nhập, sau đó chạy:

```powershell
gh auth login
gh repo create hand-gesture-slide-controller --public --source=. --remote=origin --push
```

Lệnh này cần bạn tự xác thực GitHub; mình không tự lấy hoặc lưu token của bạn.

## Lưu ý

- `models/` và dữ liệu thu thập bị ignore khỏi Git để không đẩy file lớn/dữ liệu camera.
- `src/run_gate_realtime.py` hiện in sự kiện `NEXT SLIDE`, `PREVIOUS SLIDE`, `PLAY/PAUSE`; có thể nối thêm `pyautogui` hoặc PowerPoint COM sau khi model ổn định.
- Webcam và MediaPipe cần môi trường desktop có quyền camera; không thể xác nhận hình ảnh realtime trong terminal tự động.
