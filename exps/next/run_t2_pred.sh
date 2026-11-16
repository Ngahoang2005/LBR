#!/bin/bash
# McNemar z vs f tren quy dao chi L_trunk (T2, A_L 64), 3 seed: train lai co checkpoint -> B3 luu du doan task cuoi.
cd /home/tahuuloc/Documents/httn/RSIAT
PY=/home/tahuuloc/miniconda3/envs/RSIAT/bin/python
CK=exps/claims/ckpt
for s in 1993 1994 1995; do
  tag=b4_T2_a64_s$s; cfg=exps/lbr/imageneta_$tag.json
  [ -f $CK/$tag/task_9.pt ] && [ -f $CK/$tag/task_0.pt ] || NEED=9500 bash exps/lbr/run_sweep.sh imageneta_$tag
  for k in 1 2 3 4 5; do
    [ -f exps/claims/b3_${tag}_pred.npz ] && break
    while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -lt 9500 ]; do sleep 60; done
    $PY exps/claims/b3_memory.py --config $cfg --ckpt_dir $CK/$tag --tag $tag > exps/claims/b3_$tag.out 2>&1
  done
  echo "[$(date '+%F %T')] xong $tag (pred: $(ls exps/claims/b3_${tag}_pred.npz 2>/dev/null))"
done
