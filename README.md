# Hand Gesture Slide Controller

Project HCI điều khiển slide bằng cử chỉ tay: video → MediaPipe hand landmarks → GRU sequence classifier → gate chống trigger lặp.

## Cài đặt

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn activate, chạy trực tiếp `.venv\Scripts\python.exe` hoặc đổi execution policy theo chính sách máy của bạn.

## Jester + GRU

Jester phù hợp hơn cho gesture động. Tải `Jester_zip1`, `Jester_zip2`, `Jester_zip3` và file `Labels`; ba file zip đều chứa các phần video khác nhau, không tách theo nhãn. Sau khi giải nén, lọc bốn nhãn cần dùng và tạo chuỗi landmarks:

```powershell
python src/07_prepare_jester.py --video-root <thu-muc-video-jester> --labels <Labels> --output data/jester_sequences.npz --max-per-label 1000
python src/04_train_gru.py --data data/jester_sequences.npz --model models/gru.keras
python src/05_realtime_gru.py --model models/gru.keras
```

Mapping nhãn: `Swiping Right` → `next`, `Swiping Left` → `previous`, `Zooming In` → `zoom_in`, `Zooming Out` → `zoom_out`. Có thể giữ `Doing other things` → `none` làm lớp nền để giảm false trigger. GRU dùng 32 frame và tối đa hai bàn tay.

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
