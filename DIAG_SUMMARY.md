# Tóm tắt chẩn đoán RSIAT trên ImageNet-A (B0 Inc20, ViT-B/16-IN21K)

Baseline RSIAT gốc (seed 1993, torch_seed 1): **Avg 74.76 / Last 65.77**.
Cùng code, torch_seed khác: Last 64.52–64.85 → nhiễu giữa seed ≈ ±1 điểm Last. Chênh < 1 trên 1 seed không kết luận được.
Joint (1 task, 200 lớp, cùng công thức task 0): 69.78.

## 1. Lỗi đến từ đâu (task cuối, ~35% lỗi)
- **71% số lỗi là "lỗi cứng"**: sai ở mọi backbone (R0f, joint, adapter đóng băng sau task 0), dùng classifier oracle. Không sửa được bằng cơ chế CIL.
- 16% lỗi do head/CA (oracle LDA trên cùng feature đoán đúng), 10% là khoảng cách tuần tự so với joint.
- Lỗi phụ thuộc mạnh vào số ảnh train/lớp: lớp 2–14 ảnh sai 60%, lớp 35–86 ảnh sai 25%. Logit adjustment không giúp.
- Không có thông tin bổ sung trong patch token, CLS lớp giữa, hay TTA lật ảnh.

## 2. Những gì KHÔNG phải nút thắt
| Giả thuyết | Kết quả đo (2 seed) |
|---|---|
| Drift / stats lưu lỗi thời | Thay mean+cov bằng stats thật qua model hiện tại: ≤ +0.5 |
| Replay Gaussian sai hình dạng | Train head bằng feature thật thay Gaussian: +0.14–0.16 |
| Forgetting | Với tập đối thủ cố định: +0.00 ± 0.86 (không có forgetting) |
| Thiếu plasticity | Lớp học muộn ≈ joint |
| Representation của SSRI / RAE / λ=0 khác nhau | Oracle LDA chênh ≤ 1.8; regularizer chỉ giữ stats hợp lệ, không làm feature tốt hơn |

## 3. Cơ chế còn lại: crowding theo thứ tự học
- Toàn bộ khoảng cách 2.3–2.9 điểm so với joint là nhầm **giữa các task**: lớp học sau đặt chồng lên lớp học trước trong representation.
- Mọi cách đẩy dựa trên anchor đều thất bại: L_orth (không có gradient vào adapter), OAS, full-softmax với anchor head/prototype, anchor oracle. CA xóa tác dụng ở head; ở representation không đổi.
- Thấy 20 ảnh thật mỗi lớp cũ ở task cuối: Last +1.74 (nhờ head) nhưng oracle LDA chỉ +0.10.
- Ghép feature T0+T9 hoặc nội suy trọng số θ_T0→θ_T9: chỉ là đánh đổi, không có điểm tốt hơn R0f.

## 4. Task 0 và chuyển giao sang lớp tương lai
- PTM gốc 57.4 → adapter task 0 65.6 → joint 69.1 (LDA 180 lớp tương lai). Adapter task 0 có ích cho lớp chưa thấy.
- Đường cong theo epoch là chữ U ngược: 5ep 61.0, 10ep 63.8, **20ep 65.6 (đỉnh, = mặc định)**, 40ep 62.8. Không có dư địa từ việc chọn epoch.
- Pha 2 (train quá lâu): phương sai trong lớp co lại ở toàn không gian, không phải "nhốt" vào 20 hướng head base.
- Hiện tượng chỉ có ở task 0; task 5 với 0.5×/1×/2× epoch không đổi.
- L_align cần thiết: bỏ đi (β=0) Last −2.0 đến −2.6, chủ yếu do stats lỗi thời, representation chỉ −0.7.
- RAE trong code gốc: decoder kết thúc bằng Sigmoid, không khởi tạo về 0, nên P(f) ≈ f + hằng dương; dự đoán drift kém hơn identity (~5.8×).

## 5. RFA (Rate-Floor Adaptation) — kết luận: không vượt baseline
- Kill test task 0 (40 epoch, 2 seed): mọi ρ đều không đạt 66.3, không thắng SWA (64.58). RFA giữ được độ rộng (erank 454 vs 423) nhưng transfer không giữ → nén là dấu hiệu, không phải nguyên nhân.
- Coding rate trên batch n=48 ≪ d=768 chỉ đo độ lớn phần dư, không đo số chiều; ρ ≤ 0.95 không bao giờ kích hoạt.
- Full sequence, seed gốc, 10 cấu hình (ρ, λ, epoch, cửa sổ kích hoạt): **head Last luôn thấp hơn gốc (−0.66 đến −1.52)**. Avg chỉ nhỉnh hơn khi train task 0 25–30 epoch (tối đa +0.12) nhưng Last mất nhiều hơn.
- Mẫu nhất quán: task đầu cao hơn gốc, task cuối tụt. Nguyên nhân đo được: RFA giữ lớp rộng hơn, adapter kế thừa độ rộng đó sang các task sau, lớp học muộn chồng lấn nhiều hơn khi số lớp lớn (lớp cũ ở task 6–9: −0.6 đến −0.9; drift không tăng).

## 6. Thứ duy nhất tăng điểm nhất quán (nhỏ)
Ensemble ở bước đánh giá: head (sau CA) + LDA từ stats lưu (mean lớp + covariance gộp, shrink g=1), z-score rồi trung bình 0.5/0.5. Không exemplar, không tham số mới.
Last tăng ở 4/4 run (+0.33 đến +0.92); R0f seed gốc: 74.80/65.70 → **75.20/66.03**. Bật bằng `"eval_ens_lda": true`.

## Code còn lại (mặc định tắt, không đổi hành vi RSIAT gốc)
- `models/RSIAT_adapter.py`: `eval_ens_lda`, `rfa_*`, `inc_compact_w`, `full_softmax`, `gate_old_k`, `diag_*`.
- `models/rfa.py`: RFA. `models/diag_dump.py`: dump feature/stats để chẩn đoán.
- `trainer.py`: `torch_seed`, `resume_task`, `stop_after_task`.
