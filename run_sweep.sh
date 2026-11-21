#!/bin/bash
# Evaluasi zero-shot checkpoint ToMMeR rilis satu per satu (bisa dilanjutkan jika terputus)
source ~/tommer_env.sh
cd $REPO
CKPTS=(
  roberta-base_L5_R64 bert-base-uncased_L5_R64 ModernBERT-base_L5_R64
  pythia-14m_L3_R64 pythia-31m_L1_R64 pythia-70m_L1_R64 pythia-160m_L5_R64 pythia-410m_L5_R64
  pythia-1b_L5_R64 pythia-1.4b_L5_R64 pythia-2.8b_L5_R64 phi-1_5_L5_R64 phi-2_L5_R64
  pythia-6.9b_L5_R64
)
for c in "${CKPTS[@]}"; do
  out=$WORK/results/$c.json
  n=$(python -c "import json;print(len(json.load(open('$out'))))" 2>/dev/null || echo 0)
  if [ "$n" -ge 26 ]; then echo "[skip] $c sudah lengkap"; continue; fi
  echo "[$(date '+%F %T')] mulai $c | RAM $(( $(cat /sys/fs/cgroup/memory.current)/2**30 ))/16 GB"
  python eval_table1.py llm2ner/ToMMeR-$c $out > $WORK/logs/eval_$c.log 2>&1 \
    && echo "[$(date '+%F %T')] selesai $c" \
    || { echo "[$(date '+%F %T')] GAGAL $c -> lihat $WORK/logs/eval_$c.log"; tail -5 $WORK/logs/eval_$c.log; }
done
echo "SEMUA SELESAI"
