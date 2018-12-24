# Build a random pipeline with a specified number of schedules
if [ $# -ne 3 ]; then
  echo "Usage: $0 pipeline_id num_schedules num_threads"
  exit
fi

set -eu

HL_TARGET="x86-64-avx2-disable_llvm_loop_unroll-disable_llvm_loop_vectorize"
PIPELINE_ID=${1}
BATCH_SIZE=${2}
HL_NUM_THREADS=${3}
GENERATOR=./bin/random.generator
PIPELINE=random_pipeline
STAGES=$(((PIPELINE_ID % 30) + 2))

SAMPLES=${PWD}/samples
DATA_DIR=${PWD}/data

mkdir -p ${SAMPLES}
mkdir -p ${DATA_DIR}

# A batch of this many samples is built in parallel, and then
# benchmarked serially.
if [ -z ${BATCH_SIZE} ]; then
BATCH_SIZE=1
fi

# Build a single sample of the pipeline with a random schedule
make_sample() {
    D=${1}
    SEED=${2}
    FNAME=${3}
    mkdir -p ${D}
    rm -f "${D}/sample.sample"
    if [[ $D == */0 ]]; then
        # Sample 0 in each batch is best effort beam search, with no randomness
        dropout=100
        beam=50
    else
        # The other samples are random probes biased by the cost model
        dropout=50
        beam=1
    fi

    HL_PERMIT_FAILED_UNROLL=1 \
        HL_MACHINE_PARAMS=${HL_NUM_THREADS},1,1 \
        HL_SEED=${SEED} \
        HL_SCHEDULE_FILE=${D}/schedule.txt \
        HL_FEATURE_FILE=${D}/sample.sample \
        HL_WEIGHTS_DIR=${PWD}/weights \
        HL_RANDOM_DROPOUT=${dropout} \
        HL_BEAM_SIZE=${beam} \
        HL_MACHINE_PARAMS=${HL_NUM_THREADS},1,1 \
        HL_USE_MANUAL_COST_MODEL=1 \
        HL_JSON_DUMP=${D}/${SEED}.mp \
        ${GENERATOR} \
        -g ${PIPELINE} \
        -f ${FNAME} \
        -o ${D} \
        -e stmt,assembly,static_library,h \
        target=${HL_TARGET} \
        auto_schedule=true \
        seed=${SEED} \
        max_stages=${STAGES} \
        -p bin/libauto_schedule.so \
            &> ${D}/compile_log.txt

    c++ \
        -std=c++11 \
        -DHL_RUNGEN_FILTER_HEADER="\"${D}/${FNAME}.h\"" \
        -I ../../include \
        ../../tools/RunGenMain.cpp \
        ../../tools/RunGenStubs.cpp  \
        ${D}/*.a \
        -o ${D}/bench \
        -ljpeg -ldl -lpthread -lz -lpng
}

# Benchmark one of the random samples
benchmark_sample() {
    D=${1}
    SEED=${2}

    if [ ! -f ${D}/bench ]; then
        return
    fi

    HL_NUM_THREADS=${HL_NUM_THREADS} \
        ${D}/bench \
        --output_extents=estimate \
        --default_input_buffers=random:0:estimate_then_auto \
        --default_input_scalars=estimate \
        --benchmarks=all \
        --benchmark_min_time=1 \
            | tee ${D}/bench.txt

    rm ${D}/bench

    # Add the runtime, pipeline id, and schedule id to the feature file
    R=$(cut -d' ' -f8 < ${D}/bench.txt)

    if [ -z ${R} ]; then
        return
    fi

    python3 merge_bench.py --data_dir ${D} --id ${SEED} --num_stages ${STAGES} --time ${R} 

    F=$(printf "%s.mp" ${D}/${SEED})

    if [ ! -f  ${F} ]; then
        return
    fi

    cp ${F} ${DATA_DIR}/
}

i=${PIPELINE_ID}

DIR=${SAMPLES}/batch_${i}

mkdir -p ${DIR}

# Compile a batch of samples using the generator in parallel
echo Compiling ${BATCH_SIZE} samples for batch_${i}...
for ((b=0;b<${BATCH_SIZE};b++)); do
    S=$(printf "%d%02d" $i $b)
    FNAME=$(printf "%s_batch_%02d_sample_%02d" ${PIPELINE} $i $b)
    make_sample "${DIR}/${b}" $S $FNAME &
done
wait

# Benchmark them serially using rungen
for ((b=0;b<${BATCH_SIZE};b++)); do
    S=$(printf "%d%02d" $i $b)
    benchmark_sample "${DIR}/${b}" $S
done

