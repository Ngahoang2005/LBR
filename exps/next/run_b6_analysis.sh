#!/bin/bash
# B3 + B1 tren cac quy dao B6 (LBR A_L 128/256, RSIAT-dagger A_L 64/128/256).
cd /home/tahuuloc/Documents/httn/RSIAT
PY=/home/tahuuloc/miniconda3/envs/RSIAT/bin/python
for m in b6_lbr_a128 b6_lbr_a256 b6_rsiatd_a64 b6_rsiatd_a128 b6_rsiatd_a256; do
  cfg=../RSIAT_orig/exps/run/imageneta_$m.json
  [[ $m == b6_lbr* ]] && cfg=exps/lbr/imageneta_${m}_s1993.json
  echo "[$(date '+%F %T')] B3 $m"; $PY exps/claims/b3_memory.py --config $cfg --ckpt_dir exps/claims/ckpt/$m --tag $m > exps/claims/b3_$m.out 2>&1
  echo "[$(date '+%F %T')] B1 $m"; $PY exps/claims/b1_drift.py --config $cfg --ckpt_dir exps/claims/ckpt/$m --tag $m > exps/claims/b1_$m.out 2>&1
done
echo "[$(date '+%F %T')] xong"
