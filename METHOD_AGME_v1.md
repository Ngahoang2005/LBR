# AGME v1: CA + LDA + AGM, ghép z-score

Cấu hình chốt của phương pháp. Xây trên RSIAT (class-incremental learning với ViT-B/16 IN21k và adapter).

## Định nghĩa

1. **Backbone: RSIAT-simple.** Như RSIAT gốc nhưng bỏ L_RS và L_orth (`lambda_rs = 0`, `gamma = 0`). Giữ adapter, RAE (`beta = 3`), SDC cho mean lớp cũ và CA.
2. **Ba head đánh giá trên cùng feature test** `f`:
   - **CA**: logit cosine của head sau CA (`head_logits` trong dump).
   - **LDA**: covariance chung `S` = trung bình covariance lớp, `P = (S + mean(diag S) I)^-1`, logit `f P mu_c - 0.5 mu_c P mu_c`.
   - **AGM** (không train, dạng đóng):
     - Đặc trưng φ(f) = [f ; ReLU(W f) ; 10], với W ∈ R^{4000×768}, phần tử N(0, 1/d), seed 0. Kích thước φ là 4769.
     - Mỗi lớp c: f ~ N(μ_c, Σ_c), Σ_c = covariance mẫu + 1e-3·I.
     - E[φ] và E[φφᵀ] tính **chính xác** (normal hai biến bị cắt cho khối ReLU, bổ đề Stein cho khối tuyến tính–ReLU). Không lấy mẫu.
     - Ridge: G = Σ_c n·E_c[φφᵀ], Q_c = n·E_c[φ], w_c = (G + λI)⁻¹ Q_c, trọng số lớp cân bằng n = round(mean n_c).
     - **λ cố định** = 0.1 · tr(G) / M, với M = 4769 (là `agm.M`).
3. **Dự đoán:** argmax[ z(CA) + z(LDA) + z(AGM) ], z là z-score của logit theo từng mẫu, các head có trọng số bằng nhau.

Biến thể đã thử: CA + AGM (không có LDA), gọi là z CA+AGM. Kết quả của biến thể này được ghi bên dưới để so sánh.

## Kết quả (Avg / Last, theo seed)

Số liệu lấy từ bản tính offline trước khi dọn code. Chưa chạy lại `final_eval.py` sau khi dọn.

| Dataset | Seed | z CA+LDA+AGM (chốt) | z CA+AGM | RSIAT gốc |
|---|---|---|---|---|
| ImageNet-A | 1993 | 75.39 / 65.90 | 75.25 / 65.90 | 74.76 / 65.77 |
| ImageNet-A | 1994 | 72.98 / 64.98 | 72.96 / 64.98 | 72.18 / 63.99 |
| ImageNet-A | 1995 | 72.19 / 65.70 | 72.21 / 66.10 | 71.10 / 65.31 |
| VTAB | 1993 | 94.91 / 94.86 | 94.94 / 94.86 | 94.72 / 94.69 |
| VTAB | 1994 | 94.91 / 94.84 | 94.93 / 94.91 | 94.86 / 94.74 |
| VTAB | 1995 | 94.95 / 95.01 | 94.95 / 95.02 | 94.75 / 94.80 |

Trung bình 3 seed:

| Dataset | z CA+LDA+AGM | z CA+AGM | RSIAT gốc |
|---|---|---|---|
| ImageNet-A | 73.52 / 65.53 | 73.47 / 65.66 | 72.68 / 65.02 |
| VTAB | 94.92 / 94.90 | 94.94 / 94.93 | 94.78 / 94.74 |

## Code

| File | Vai trò |
|---|---|
| `models/RSIAT_adapter.py` | Backbone RSIAT-simple: train adapter, RAE, SDC, CA, ghi dump |
| `models/base.py` | CA (`_stage2_compact_classifier`), SDC (`displacement`), thống kê lớp |
| `models/agm.py` | Head AGM: đặc trưng, moment chính xác, ridge |
| `models/diag_dump.py` | Ghi dump sau mỗi task |
| `exps/agm/final_eval.py` | Tính CA, LDA, AGM và ghép z-score trên dump; in bảng theo task |
| `trainer.py`, `main.py` | Vòng lặp task, log, đánh giá |

## Config

- ImageNet-A: `exps/simple/imageneta_RSIAT_simple.json`, `_s1994.json`, `_s1995.json`
- VTAB: `exps/multi/vtab_simple_s1993.json`, `_s1994.json`, `_s1995.json`
- RSIAT gốc để so sánh: `exps/adapter_imageneta.json` (seed 1993), `exps/simple/imageneta_RSIAT_orig_s1994.json`, `_s1995.json`; `exps/multi/vtab_orig_s199{3,4,5}.json`

## Chạy lại

```bash
bash args.sh                    # 6 lệnh train, ghi dump (xem args.sh)
python exps/agm/final_eval.py ina_1993=exps/simple/dump/imageneta_RSIAT_simple \
  ina_1994=exps/simple/dump/imageneta_RSIAT_simple_s1994 \
  ina_1995=exps/simple/dump/imageneta_RSIAT_simple_s1995 \
  vtab_1993=exps/multi/dump/vtab_simple_s1993 \
  vtab_1994=exps/multi/dump/vtab_simple_s1994 \
  vtab_1995=exps/multi/dump/vtab_simple_s1995
```

Kết quả từng dataset và seed được ghi vào `exps/agm/results/<tên>.json`.

## Hạn chế

- AGM chỉ được tính offline trên dump. Lúc train, code không dùng AGM và không ghép z-score.
- Chưa thử IN-R, CIFAR. CUB đã bỏ.
- Ensemble hiệu chỉnh nhiệt độ hỏng trên VTAB khi có LDA, nên không dùng.
