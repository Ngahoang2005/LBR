#!/bin/bash
# Doi beta10 xong roi dung hang doi beta-sweep (bo 0.3/6 dang cho) va chay beta 5 -> 0.3.
cd /home/tahuuloc/Documents/httn/RSIAT
until grep -q "lbr_beta10_s1993 \(Avg\|KHONG\)" exps/lbr/queue_beta.txt; do sleep 20; done
systemctl --user stop beta-sweep
sleep 5
rm -f logs/adapter/imageneta/0/20/lbr_beta0p3_s1993_*
echo "[$(date '+%F %T')] doi thu tu: beta 5 truoc"
NEED=9500 bash exps/lbr/run_sweep.sh imageneta_lbr_beta5_s1993 imageneta_lbr_beta0p3_s1993
