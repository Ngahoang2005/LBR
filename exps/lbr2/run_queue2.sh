#!/bin/bash
# Chay lan luot cac config; doi GPU con >= NEED MiB truoc khi bat dau; run khong xong (OOM / bi giet) thi chay lai.
PY=/home/tahuuloc/miniconda3/envs/RSIAT/bin/python
NEED=${NEED:-14000}
TRIES=${TRIES:-5}
for name in "$@"; do
  log=$($PY -c "
import json; c = json.load(open('exps/lbr2/$name.json'))
init = 0 if c['init_cls'] == c['increment'] else c['init_cls']
print(f\"logs/{c['model_name']}/{c['dataset']}/{init}/{c['increment']}/{c['prefix']}_{c['seed'][0]}_{c['convnet_type']}.log\")")
  for try in $(seq 1 $TRIES); do
    if [ -f "$log" ] && grep -q "Average Seed Accuracy" "$log"; then echo "[skip] $name"; break; fi
    [ -f "$log" ] && mv "$log" "$log.incomplete_$(date +%m%d%H%M%S)"
    while :; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      [ "$free" -ge "$NEED" ] && break
      sleep 60
    done
    echo "[$(date '+%F %T')] start $name (lan $try, GPU trong $free MiB)"
    $PY rsiat_run.py --config exps/lbr2/$name.json > exps/lbr2/out/$name.out 2>&1
    rc=$?
    if grep -q "Average Seed Accuracy" "$log" 2>/dev/null; then echo "[$(date '+%F %T')] end $name (exit $rc, ok)"; break; fi
    why=$(grep -q "OutOfMemoryError" exps/lbr2/out/$name.out && echo OOM || echo "bi dung/loi")
    echo "[$(date '+%F %T')] end $name (exit $rc, KHONG_XONG: $why)"
    sleep 30
  done
done
