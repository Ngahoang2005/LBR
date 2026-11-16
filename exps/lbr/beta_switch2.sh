#!/bin/bash
# Doi beta5 xong roi dung beta-sweep2 (bo 0.3 dang cho) va chay beta 2 -> 0.3.
cd /home/tahuuloc/Documents/httn/RSIAT
until grep -q "lbr_beta5_s1993 \(Avg\|KHONG\)" exps/lbr/queue_beta.txt; do sleep 20; done
systemctl --user stop beta-sweep2
sleep 5
rm -f logs/adapter/imageneta/0/20/lbr_beta0p3_s1993_*
echo "[$(date '+%F %T')] doi thu tu: beta 2 truoc"
NEED=9500 bash exps/lbr/run_sweep.sh imageneta_lbr_beta2_s1993 imageneta_lbr_beta0p3_s1993
