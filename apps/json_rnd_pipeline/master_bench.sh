#!/bin/bash

# Start a watchdog to kill any compilations that take too long
bash ./watchdog.sh &
WATCHDOG_PID=$!

function finish {
  kill $WATCHDOG_PID
}
trap finish EXIT

mkdir -p results

PIPELINES_BEGIN=0
PIPELINES_END=100
PIPELINES=100
SCHEDULES=1

# Build the shared things by building one pipeline
make bin/random_pipeline.generator
#HL_TARGET=host-new_autoscheduler HL_SEED=root PIPELINE_SEED=0 HL_RANDOM_DROPOUT=100 HL_BEAM_SIZE=20 make build

for ((b=0;b<1;b++)); do
  echo Batch $b
  rm -f results/files_${b}.txt
  rm -f results/files_root_${b}.txt
  rm -f results/files_master_${b}.txt
    
  # Build lots of pipelines
  for ((p=${PIPELINES_BEGIN};p<${PIPELINES_END};p++)); do
    P=$((b * $PIPELINES + p))
    STAGES=$(((P % 30) + 10))
    echo echo Building pipeline $P
    echo "HL_TARGET=host-new_autoscheduler HL_SEED=root PIPELINE_SEED=$P PIPELINE_STAGES=$STAGES HL_RANDOM_DROPOUT=100 HL_BEAM_SIZE=20 make build 2>&1 | grep -v Nothing.to.be.done"
    echo "HL_TARGET=host HL_SEED=master PIPELINE_SEED=$P PIPELINE_STAGES=$STAGES HL_RANDOM_DROPOUT=100 HL_BEAM_SIZE=20 make build 2>&1 | grep -v Nothing.to.be.done"        
    for ((s=0;s<$SCHEDULES;s++)); do
      echo "HL_TARGET=host-new_autoscheduler HL_SEED=$s PIPELINE_SEED=$P PIPELINE_STAGES=$STAGES HL_RANDOM_DROPOUT=100 HL_BEAM_SIZE=20 make build 2>&1 | grep -v Nothing.to.be.done"
    done
  done | xargs -P8 -I{} bash -c "{}"

  # Benchmark them
  for ((p=${PIPELINES_BEGIN};p<${PIPELINES_END};p++)); do
    P=$((b * $PIPELINES + p))
    STAGES=$(((P % 30) + 10))
    echo Benchmarking pipeline $P

    F=bin/host-new_autoscheduler/pipeline_${P}_${STAGES}/schedule_root_100_20/times.txt
    if [ ! -f $F ]; then HL_TARGET=host-new_autoscheduler HL_SEED=root PIPELINE_SEED=$P PIPELINE_STAGES=$STAGES HL_RANDOM_DROPOUT=100 HL_BEAM_SIZE=20 HL_NUM_THREADS=8 make bench 2>&1 | grep -v "Nothing to be done"; fi
    grep '^Time' $F > /dev/null && echo $F >> results/files_root_${b}.txt

    F=bin/host/pipeline_${P}_${STAGES}/schedule_master_100_20/times.txt
    if [ ! -f $F ]; then HL_TARGET=host HL_SEED=master PIPELINE_SEED=$P PIPELINE_STAGES=$STAGES HL_RANDOM_DROPOUT=100 HL_BEAM_SIZE=20 HL_NUM_THREADS=8 make bench 2>&1 | grep -v "Nothing to be done"; fi
    grep '^Time' $F > /dev/null && echo $F >> results/files_master_${b}.txt        

    for ((s=0;s<$SCHEDULES;s++)); do
      F=bin/host-new_autoscheduler/pipeline_${P}_${STAGES}/schedule_${s}_100_20/times.txt
      if [ ! -f $F ]; then HL_TARGET=host-new_autoscheduler HL_SEED=$s PIPELINE_SEED=$P PIPELINE_STAGES=$STAGES HL_RANDOM_DROPOUT=100 HL_BEAM_SIZE=20 HL_NUM_THREADS=8 make bench 2>&1 | grep -v "Nothing to be done"; fi

      grep '^Time' $F > /dev/null && echo $F >> results/files_${b}.txt
    done
  done

  # Extract the runtimes
  echo "Extracting runtimes..."
  cat results/files_${b}.txt | while read F; do grep '^Time' $F | cut -d: -f2 | cut -b2-; done > results/runtimes_${b}.txt

  # Extract the number of malloc calls
  echo "Extracting mallocs..."
  cat results/files_${b}.txt | while read F; do grep '^Malloc' $F | cut -d: -f2 | cut -b2-; done > results/mallocs_${b}.txt    

  # Extract the compute_root runtimes
  echo "Extracting compute_root runtimes..."
  cat results/files_root_${b}.txt | while read F; do grep '^Time' $F | cut -d: -f2 | cut -b2-; done > results/root_runtimes_${b}.txt

  # Extract the master runtimes
  echo "Extracting master runtimes..."
  cat results/files_master_${b}.txt | while read F; do grep '^Time' $F | cut -d: -f2 | cut -b2-; done > results/master_runtimes_${b}.txt    

  # Extract the features
  echo "Extracting features..."
  cat results/files_${b}.txt | while read F; do echo $(grep '^YYY' ${F/times/stderr} | cut -d' ' -f3-); done > results/features_${b}.txt

  # Extract failed proofs
  echo "Extracting any failed proofs..."
  cat results/files_${b}.txt | while read F; do grep -A1 'Failed to prove' ${F/times/stderr}; done > results/failed_proofs_${b}.txt    

  # Extract the cost according to the hand-designed model (just the sum of a few of the features)
  echo "Extracting costs..."
  cat results/files_${b}.txt | while read F; do echo $(grep '^State with cost' ${F/times/stderr} | cut -d' ' -f4 | cut -d: -f1); done  > results/costs_${b}.txt
done

