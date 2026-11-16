#!/bin/bash
# B6 (fine-tune MLP_L, 1 seed) roi B4 factorial (T1..T4 x +-L_orth x A_L {64,256} x 3 seed). Moi run: train co
# checkpoint -> B3 + B1 -> xoa checkpoint trung gian (giu task_9.pt; thu muc la symlink toi run da co thi giu nguyen).
cd /home/tahuuloc/Documents/httn/RSIAT
PY=/home/tahuuloc/miniconda3/envs/RSIAT/bin/python
CK=exps/claims/ckpt
analyse() {   # $1 tag, $2 config
  [ -f exps/claims/b3_$1.json ] || $PY exps/claims/b3_memory.py --config $2 --ckpt_dir $CK/$1 --tag $1 > exps/claims/b3_$1.out 2>&1
  [ -f exps/claims/b1_$1.json ] || $PY exps/claims/b1_drift.py --config $2 --ckpt_dir $CK/$1 --tag $1 > exps/claims/b1_$1.out 2>&1
  [ -L $CK/$1 ] || find $CK/$1 -name 'task_*.pt' ! -name 'task_9.pt' -delete
  echo "[$(date '+%F %T')] phan tich xong $1"
}
NEED=11000 bash exps/lbr/run_sweep.sh imageneta_b6_lbr_mlp_s1993
analyse b6_lbr_mlp_s1993 exps/lbr/imageneta_b6_lbr_mlp_s1993.json
(cd ../RSIAT_orig && NEED=11000 bash run_queue.sh imageneta_b6_rsiatd_mlp)
analyse b6_rsiatd_mlp ../RSIAT_orig/exps/run/imageneta_b6_rsiatd_mlp.json
while read tag; do
  if [ ! -f $CK/$tag/task_9.pt ]; then NEED=11000 bash exps/lbr/run_sweep.sh imageneta_$tag; fi
  analyse $tag exps/lbr/imageneta_$tag.json
done < exps/next/b4_names.txt
echo "[$(date '+%F %T')] xong B4"
