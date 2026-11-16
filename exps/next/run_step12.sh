#!/bin/bash
# Buoc 1-2: RSIAT + CA s2 + shrinkage tai f (confounder), roi 3 class order cho RSIAT, RSIAT+s2+shrink, LBR.
cd /home/tahuuloc/Documents/httn/RSIAT_orig
NEED=11000 bash run_queue.sh imageneta_rsiat_s2shrink_s1993 imageneta_rsiat_s1994 imageneta_rsiat_s2shrink_s1994 imageneta_rsiat_s1995 imageneta_rsiat_s2shrink_s1995
cd /home/tahuuloc/Documents/httn/RSIAT
NEED=11000 bash exps/lbr/run_sweep.sh imageneta_lbr_split_ae_md1_s1994 imageneta_lbr_split_ae_md1_s1995
echo "[$(date '+%F %T')] xong buoc 1-2"
