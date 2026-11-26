# HCI-project — repo nghiên cứu (train model, mô phỏng, đánh giá policy)

Nhóm 12, chủ đề B2: điều khiển slide bằng cử chỉ tay, chống Midas Touch (kích hoạt nhầm khi thuyết trình).
Repo này **train và đánh giá offline**. App chạy thật nằm ở repo riêng `HCI-Slideshow-Interaction`
(`C:\Users\admin\OneDrive\Máy tính\HCI\HCI-Slideshow-Interaction`, có CLAUDE.md riêng); app chỉ **dùng** model copy từ đây.
Mục tiêu đề tài (RQ1 ngưỡng thích ứng, RQ2 abstention, 4 cấu hình C1–C4, chỉ số) ghi đầy đủ trong CLAUDE.md của app.

## Quy ước làm việc

- **Không tự commit / merge / push.** Người dùng tự commit; chỉ sửa file và để nguyên chưa commit.
- Nhiều file đang có thay đổi chưa commit của thành viên khác (`src/04_train_gru.py`, `05_realtime_gru.py`, `jester_data.py`,
  `tests/test_jester_data.py`, model). Sửa tối thiểu, ưu tiên thêm file mới.
- Console Windows dùng cp1252: script in tiếng Việt phải `sys.stdout.reconfigure(encoding="utf-8")` hoặc chạy với `PYTHONIOENCODING=utf-8`.
- Đa tiến trình (`ProcessPoolExecutor`) chỉ chạy từ file `.py`, không chạy từ `python - <<EOF` (Windows spawn cần đường dẫn script).
- Test: `python -m pytest -q tests` (42 test, đều pass).
- Train trên Windows chỉ chạy CPU (TF ≥ 2.11 không có GPU trên Windows). GPU (RTX 3050 4 GB) dùng qua WSL2 Ubuntu, đã cài 2026-11-25:
  `/root/hci-gpu` (Python 3.12, `tensorflow[and-cuda]==2.16.2`, keras 3.15.1). Chạy: `wsl -d Ubuntu`, `source ~/hci-gpu/bin/activate`,
  `export TF_FORCE_GPU_ALLOW_GROWTH=true`, rồi gọi script như thường. Trong WSL, `infer_ms` đo trên GPU (app chạy CPU) -> đo lại trên Windows.
  Không trộn model train CPU và GPU trong một bảng so sánh.

## Cấu trúc

| File | Vai trò |
|---|---|
| `src/sequence_utils.py` | Tiền xử lý chuẩn: 63 tọa độ thô + 7 motion (`motion_sequence`), `sliding_windows` (đệm 0 bên trái). App sao chép y hệt — đổi ở đây phải đổi app + sinh lại fixture parity của app. |
| `src/jester_data.py` | Nạp CSV landmark (Jester hoặc dữ liệu tự thu) thành `VideoSet`, dựng luồng giả lập (`build_streams`, `stream_windows`). |
| `src/04_train_gru.py` | Train GRU gốc 5 lớp (`next, none, previous, zoom_in, zoom_out`) + `build_training_set` (aligned + dịch thời gian/jitter + mẫu "tay giữ yên" = none). |
| `src/06_simulate_stream.py` | Mô phỏng luồng camera trên Jester -> log dự đoán từng khung. |
| `src/policies.py` | C1–C4 (`DecisionPolicy`). C2 = công thức ngưỡng thích ứng của app (đã đối chiếu lệch 0). Abstention mới (2026-11-21) cùng thiết kế app. `reverse_block_frames`, `repeat_block_frames` = heuristic của app, mặc định tắt. |
| `src/evaluate_policies.py` | Chạy log dự đoán qua policy theo trial. `--commands next,previous` gộp zoom vào none như app. `prepare_trial`/`evaluate_prepared` để chạy nhanh nhiều cấu hình. |
| `src/model_zoo.py` | 7 kiến trúc: gru, gru_small, lstm, bigru, tcn, attention, mlp (chỉ layer Keras chuẩn để app nạp được). |
| `src/09_compare_models.py` | Train mọi kiến trúc cùng dữ liệu/seed, báo cáo phân loại, tốc độ, mô phỏng val/test -> `models/zoo/`, `data/zoo_predictions/`. Cache CSV 490 MB ở `data/cache/`. |
| `src/10_tune_policies.py` | Quét ~2.750 cấu hình C1–C4 trên validation, chọn theo ngân sách kích hoạt nhầm (1/2/3%) và theo độ trễ, test 1 lần, bootstrap CI -> `reports/tuning/`. |
| `src/11_summarize_experiments.py` | -> `reports/EXPERIMENTS_RESULTS.md`, `reports/tradeoff_policies.png`. |
| `src/scale_augment.py` | `shrink_tracks`: co tọa độ tay 1..k lần + dịch vị trí (giả lập đứng xa) cho `--far-scale`. |
| `src/13_sweep_real_threshold.py` | Quét ngưỡng × giữ (C1) trên dữ liệu tự thu cho danh sách model. |
| `src/policy_variants.py` | `VariantPolicy`: C1–C4 mở rộng để thí nghiệm, mặc định giống hệt `policies.py`. |
| `src/14_policy_experiments.py` | Thí nghiệm tham số C2/C3 trên phiên Jester dài + dữ liệu thật (`predict` rồi `run`; `run` ghi đè CSV). |
| `src/12_eval_real.py` | Đánh giá model trên **dữ liệu tự thu**: 6 video dài chạy như luồng liên tục + 61 clip near-miss -> `reports/real/`. |
| `src/05_realtime_gru.py` | Demo realtime cũ của repo (lấy 1/4 khung = 7.5 fps — xem "Vấn đề đã biết"). |
| `src/01_*, 02_*, 07_*, 08_jester.py, organize_jester.py` | Webcam, vẽ landmark, chuẩn bị/trích Jester (`08_jester.py --step 2` -> 6 fps). |

Dữ liệu (`data/`):
- `jester_landmarks_frames.csv` (490 MB): Jester theo frame, 6 fps, 58k video, cột `split` train/validation/test. Dùng cho model hiện tại.
- `jester_landmarks.csv`: bản cũ 1 dòng/video (model v0, đã bỏ).
- **Dữ liệu tự thu** (`real_*.csv`, mô tả trong `data/REAL_DATA_README.md`): 67 video quay thật của 2 người, 1280x720, đã bỏ lật gương
  về hệ tọa độ Jester, **lấy mẫu 7.5 fps**. `real_short_landmarks.csv` = 61 clip near-miss ngắn; `real_long_landmarks.csv` = 6 video thuyết trình
  dài liên tục (~18.5 phút, 24 next + 5 previous, 21 đoạn near-miss, 53 đoạn nền; nhãn đã kéo dài +1.5 s ở đuôi); `real_clips_landmarks.csv` =
  164 clip gộp dạng Jester. Toàn bộ `split = test`. `*_annotations-v5.csv` = nhãn thủ công gốc.
- `data/cache/`, `data/zoo_predictions/`: sinh tự động, đã gitignore.

Model: `models/gru.keras` (GRU v1, model gốc app từng dùng), `models/zoo/<kiến trúc>_L<cửa sổ>.keras` (+ `.labels.json`, `.meta.json`).
Copy sang app: 3 file vào `HCI-Slideshow-Interaction/models/gru/` với tên `gru.*`, rồi sinh lại fixture parity của app.
**App hiện dùng `bigru_L16`** (từ 2026-11-21).

## Bất biến (sai là model chạy sai mà không báo lỗi)

1. Đặc trưng = 63 tọa độ ảnh **thô** (không chuẩn hóa theo kích thước tay) + 7 motion so với bước trước; không tay -> 0; motion = 0 khi
   bước này hoặc bước trước không có tay. Vì tọa độ thô nên model **phụ thuộc kích thước tay trong khung hình** (xem kết quả dữ liệu thật).
2. Nhịp lấy mẫu phải khớp `step_ms` trong `.meta.json` (6 fps = 166.7 ms cho mọi model hiện tại).
3. Hệ tọa độ = ảnh **không lật gương** như Jester: `next` (Swiping Right) có dx cổ tay < 0. App lật camera rồi lật lại x (`gru_flip_x`).
   Dữ liệu tự thu đã được bỏ lật gương lúc trích; kiểm chứng: lật thêm lần nữa thì người 1 từ 6/9 lệnh đúng xuống 0/9.
4. `jester_data.load_videos`: CSV có cột `scenario` + `label` (dữ liệu tự thu) dùng nguyên nhãn đó; CSV Jester (không có `scenario`) suy nhãn từ `raw_label`.

## Kết quả đã có

**Jester (mô phỏng, 2026-11-21)** — chi tiết `reports/EXPERIMENTS_NOTES.md`, `reports/EXPERIMENTS_RESULTS.md`, và tài liệu
"Báo cáo thực nghiệm: so sánh model và chỉnh tham số chống Midas Touch" (https://claude.ai/artifact/R2SvT4yuVfysuULByisQFm):
- Phân loại: GRU 92.2%, BiGRU 92.2%, TCN 91.9%, LSTM 91.8% ... MLP 88.6%. Một seed/model.
- Sau chỉnh tham số (kích hoạt nhầm ≤ 2%): BiGRU+C2 78.5% đúng lệnh sạch, LSTM+C4 77.7%, GRU+C1 75.3%. Tham số chọn: ngưỡng ~0.8, giữ 500 ms, cooldown 0.8 s.
- RQ1: C2 ≈ C1 ở 7/9 model, hơn ở BiGRU/TCN. RQ2: abstention không nhất quán.
- Mô phỏng khác thực tế: giữ 500 ms tối ưu trên Jester nhưng trên webcam thật làm sót ~6/10 cú vuốt (app dùng giữ 100 ms).

**Dữ liệu tự thu (2026-11-22)** — phân tích `reports/REAL_DATA_EVAL.md`, số liệu `reports/real/summary.csv`:
- Mọi model Jester nhận kém trên 6 video thuyết trình: tốt nhất 11/29 lệnh (GRU cửa sổ 10, nhận ngay), BiGRU 7/29; người 1 4–7/9, người 2 0–5/20.
- **Nguyên nhân chính: người 2 đứng xa** — video 1–3 tay nhỏ ~4 lần Jester, vuốt quét ~8% khung -> 0/12 lệnh. Tọa độ thô nên model không bất biến tỉ lệ.
  Phóng to lúc chạy (`--zoom`) nhận gấp đôi nhưng nhầm tăng 17–36 lần -> phải xử lý khi train (tăng cường tỉ lệ / chuẩn hóa / fine-tune dữ liệu thật).
- C1–C4 chênh ≤ 2 lệnh trên dữ liệu thật; giữ lâu hơn chỉ giảm số lệnh trúng. 7.5 vs 6 fps chênh ≤ 3 lệnh, không phải nguyên nhân chính.
- Near-miss ngắn bị bắn 23–34/61 clip; cự ly gần 69% vs xa 12% (BiGRU). Nhầm khi nói bình thường thấp: 0–8 lần / 17 phút.
- Clip "thực hiện lệnh rồi thu tay về" (21 clip) đang gán `none` dù chứa lệnh thật -> nên gán lại "lệnh X + cấm lệnh ngược".
- Video 5 của người 2: nhãn next bị model đọc là previous ~1.0 (lật x chỉ sửa 2/4) -> xem lại video gốc.
- **Tăng cường "đứng xa" đã thử (mục 6 của REAL_DATA_EVAL.md):** `src/scale_augment.py` + `09_compare_models.py --far-scale 5 --tag far`
  -> `models/zoo/*_far.keras`. Trên Jester gần bất biến tỉ lệ (×4: 90% so với 49–60% của model cũ). Trên dữ liệu thật trúng 6–11 -> 11–16/29
  (người 2: 2 -> 6–9/20) nhưng nhầm 2–4 -> 87–133 lần/17 phút (chủ yếu video xa của người 2). Ở cùng mức nhầm thì **không hơn model cũ**
  (quét ngưỡng: `src/13_sweep_real_threshold.py`). App vẫn dùng `bigru_L16`. Bước tiếp: fine-tune với cử động nói thật ở xa làm lớp none.

**Tham số C2/C3 (2026-11-25, 5 model)** — `reports/POLICY_PARAM_EXPERIMENTS.md`: cửa sổ ổn định và dải ngưỡng C2 không có tác dụng (lịch sử bị xóa);
C2 gốc không hơn C1; C3 hơn C1 1.6–4.3 điểm trên Jester, nhưng "dung sai khung rớt" giúp C1 xấp xỉ C3 ở 3/5 model. Trên pipeline app + dữ liệu thật
chưa thấy khác biệt; `clear_conf` cao (tốt trên Jester) làm sót vuốt ở session webcam.

## Vấn đề đã biết

- **Lệch nhịp 7.5 fps / 6 fps:** dữ liệu tự thu và `05_realtime_gru.py` lấy 1/4 khung (7.5 fps, theo giả định cũ của model v0),
  còn model v1 và mọi model trong `models/zoo` train ở 6 fps. `12_eval_real.py --rates native,6fps` đo cả hai; app thì lấy mẫu đúng 6 fps.
- `data/REAL_DATA_README.md` nhắc `real_long_annotations.csv`, `real_predictions.csv`, `src/09_extract_real_videos.py` nhưng các file này
  **chưa có trong repo**. "Cách A" của README cần `jester_data.py` đọc được cột `scenario` (đã sửa 2026-11-22).
- `harmonic_oscillation_annotations-v5.csv` (364 s) và `harmonic_oscillation_part2_annotations-v5.csv` (640 s) gán nhãn thêm ~17 phút video
  (18 next, 3 prev, 32 near-miss) nhưng **chưa được trích landmark**.
- `reports/` nằm trong `.gitignore`: muốn lưu kết quả vào git thì `git add -f`.

## Lệnh thường dùng

```powershell
python src/09_compare_models.py --runs gru:16,bigru:16      # train + mô phỏng (≈5–20 phút/model trên CPU)
python src/10_tune_policies.py --runs gru_L16 --reuse-grid   # chọn tham số C1-C4 (≈4 phút/model với 18 worker)
python src/11_summarize_experiments.py
python src/12_eval_real.py --runs bigru_L16 --rates 6fps     # đánh giá trên dữ liệu tự thu (≈1.5 phút/model/nhịp)
python -m pytest -q tests
```
