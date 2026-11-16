#!/bin/bash
# B6: LBR va RSIAT-dagger (RSIAT + CA s2 + shrink tai f) voi bottleneck A_L 128, 256 (+ RSIAT-dagger 64), co checkpoint.
cd /home/tahuuloc/Documents/httn/RSIAT
NEED=11000 bash exps/lbr/run_sweep.sh imageneta_b6_lbr_a128_s1993 imageneta_b6_lbr_a256_s1993
cd ../RSIAT_orig
NEED=11000 bash run_queue.sh imageneta_b6_rsiatd_a64 imageneta_b6_rsiatd_a128 imageneta_b6_rsiatd_a256
echo "[$(date '+%F %T')] xong B6"
