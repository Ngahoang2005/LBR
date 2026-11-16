#!/bin/bash
# Nhom 1-3: doi hang doi B4 cu xong run dang chay roi dung no; smoke test tuy chon moi; chay lan luot
# exps/next/groups_queue.txt ("lbr|orig tag"), moi run: train co checkpoint -> B3 + B1 -> xoa checkpoint (giu task_9).
cd /home/tahuuloc/Documents/httn/RSIAT
PY=/home/tahuuloc/miniconda3/envs/RSIAT/bin/python
CK=exps/claims/ckpt
# (hang doi B4 cu da dung, smoke test da qua luc 11:08 15/11)
while read repo tag; do
  if [ $repo = lbr ]; then cfg=exps/lbr/imageneta_$tag.json
    [ -f $CK/$tag/task_9.pt ] || NEED=${NEED:-9500} bash exps/lbr/run_sweep.sh imageneta_$tag
  else cfg=../RSIAT_orig/exps/run/imageneta_$tag.json
    [ -f $CK/$tag/task_9.pt ] || (cd ../RSIAT_orig && NEED=${NEED:-9500} bash run_queue.sh imageneta_$tag)
  fi
  [ -f exps/claims/b3_$tag.json ] || $PY exps/claims/b3_memory.py --config $cfg --ckpt_dir $CK/$tag --tag $tag > exps/claims/b3_$tag.out 2>&1
  [ -f exps/claims/b1_$tag.json ] || $PY exps/claims/b1_drift.py --config $cfg --ckpt_dir $CK/$tag --tag $tag > exps/claims/b1_$tag.out 2>&1
  [ -L $CK/$tag ] || find $CK/$tag -name 'task_*.pt' ! -name 'task_9.pt' -delete
  echo "[$(date '+%F %T')] xong $tag"
done < exps/next/groups_queue.txt
echo "[$(date '+%F %T')] xong nhom 1-3"
