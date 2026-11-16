#!/bin/bash
cd /home/tahuuloc/Documents/httn/RSIAT
echo "[$(date '+%F %T')] tiep ablation ImageNet-A (bo A3)"
bash exps/lbr2/run_queue2.sh abl_A4_nobg_s1993 abl_Aor_oracle_s1993 abl_B0_samp_s1993 abl_B1_unit_s1993 abl_B2_pool_s1993 \
  abl_B4_n05_s1993 abl_B4_n050_s1993 abl_B4_n0100_s1993
echo "[$(date '+%F %T')] ablation xong"
