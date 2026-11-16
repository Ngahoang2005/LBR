#!/bin/bash
# B1/B2 tren ImageNet-A, moi luc mot job GPU: doi run LBR co checkpoint (unit claims-lbr) xong -> B2 tren model task 0
# cua LBR -> RSIAT goc va SSIAT (RSIAT khong L_RS, RAE, L_orth) co checkpoint moi task (../RSIAT_orig) -> B2 tren model
# task 0 cua RSIAT -> B1 cho LBR, RSIAT, SSIAT.
cd /home/tahuuloc/Documents/httn/RSIAT
PY=/home/tahuuloc/miniconda3/envs/RSIAT/bin/python
ORIG=../RSIAT_orig/exps/run
while systemctl --user is-active --quiet claims-lbr; do sleep 60; done
echo "[$(date '+%F %T')] B2 lbr"
$PY exps/claims/b2_fidelity.py --config exps/lbr/imageneta_claims_lbr_s1993.json --ckpt exps/claims/ckpt/lbr/task_0.pt \
  --tag lbr > exps/claims/b2_lbr.out 2>&1
(cd ../RSIAT_orig && NEED=11000 bash run_queue.sh imageneta_claims_rsiat imageneta_claims_ssiat)
echo "[$(date '+%F %T')] B2 rsiat"
$PY exps/claims/b2_fidelity.py --config $ORIG/imageneta_claims_rsiat.json --ckpt exps/claims/ckpt/rsiat/task_0.pt \
  --tag rsiat > exps/claims/b2_rsiat.out 2>&1
for m in lbr rsiat ssiat; do
  cfg=$ORIG/imageneta_claims_$m.json
  [ $m = lbr ] && cfg=exps/lbr/imageneta_claims_lbr_s1993.json
  echo "[$(date '+%F %T')] B1 $m"
  $PY exps/claims/b1_drift.py --config $cfg --ckpt_dir exps/claims/ckpt/$m --tag $m > exps/claims/b1_$m.out 2>&1
done
echo "[$(date '+%F %T')] xong"
