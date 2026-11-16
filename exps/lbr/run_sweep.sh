#!/bin/bash
# Chay lan luot (run_queue tu chay lai khi bi dung), in Avg/Last moi run xong.
for name in "$@"; do
  bash exps/lbr/run_queue.sh $name
  log=$(python3 -c "import json;c=json.load(open('exps/lbr/$name.json'));i=0 if c['init_cls']==c['increment'] else c['init_cls'];print(f\"logs/adapter/{c['dataset']}/{i}/{c['increment']}/\")")$(python3 -c "import json;print(json.load(open('exps/lbr/$name.json'))['prefix'])")_$(python3 -c "import json;print(json.load(open('exps/lbr/$name.json'))['seed'][0])")_pretrained_vit_b16_224_in21k_adapter.log
  grep -q "Average Seed Accuracy" $log || { echo "[$(date '+%F %T')] $name KHONG XONG"; continue; }
  echo "[$(date '+%F %T')] $name Avg $(grep -h 'Average Accuracy (CNN)' $log | tail -1 | awk '{print $NF}') Last $(grep -h 'CNN top1 curve' $log | tail -1 | sed 's/.*, //; s/]//')"
done
