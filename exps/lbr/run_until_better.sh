#!/bin/bash
# Chay lan luot cac config ImageNet-A (run_queue tu chay lai khi bi dung); chi xet run XONG du 10 task,
# dung khi Avg vuot A2 (75.565) va Last >= A2 (66.43).
for name in "$@"; do
  bash exps/lbr/run_queue.sh $name
  log=logs/adapter/imageneta/0/20/$(python3 -c "import json;print(json.load(open('exps/lbr/$name.json'))['prefix'])")_1993_pretrained_vit_b16_224_in21k_adapter.log
  grep -q "Average Seed Accuracy" $log || { echo "[$(date '+%F %T')] $name KHONG XONG"; continue; }
  avg=$(grep -h "Average Accuracy (CNN)" $log | tail -1 | awk '{print $NF}')
  last=$(grep -h "CNN top1 curve" $log | tail -1 | sed 's/.*, //; s/]//')
  echo "[$(date '+%F %T')] $name Avg $avg Last $last"
  python3 -c "import sys; sys.exit(0 if $avg > 75.565 and $last >= 66.43 else 1)" && { echo "VUOT A2 -> dung"; break; }
done
