include ../support/Makefile.inc

# TODO(mgharbi): this is a hack, the vendors header are not part of the distrib
JSON=$(HALIDE_DISTRIB_PATH)/../vendors/include

PIPELINE_SEED ?=0
PIPELINE_STAGES ?=20
HL_RANDOM_DROPOUT ?=100
HL_SEED ?=0
HL_BEAM_SIZE ?=2
HL_TARGET ?= host-new_autoscheduler
ID = $(HL_TARGET)/pipe$(PIPELINE_SEED)_stages$(PIPELINE_STAGES)/schedule_seed$(HL_SEED)_dropout$(HL_RANDOM_DROPOUT)_beam$(HL_BEAM_SIZE)

ifeq ($(HL_SEED), root)
AUTO_SCHEDULE=false
else
AUTO_SCHEDULE=true
endif

# For autoscheduler's python RPC
ZMQ_LIBS=$(shell pkg-config --libs libzmq)
LDFLAGS += $(ZMQ_LIBS) -lcurses
CXXFLAGS += -I$(JSON)

all: build

build_shared: $(BIN)/runtime.a
	@echo CXX $(CXXFLAGS)

build: $(BIN)/$(ID)/benchmark

# Runtime, shared across all random pipelines
$(BIN)/runtime.a: $(BIN)/random_pipeline.generator
	@echo "Building runtime.a"
	@$^ -r runtime target=$(HL_TARGET) -o $(BIN)

# Build the Halide generator that generates all the pipelines
$(BIN)/random_pipeline.generator: random_pipeline_generator.cpp $(GENERATOR_DEPS)
	@echo "Building random_pipeline.generator"
	@mkdir -p $(@D)
	@$(CXX) $(CXXFLAGS) -fno-rtti $(filter-out %.h,$^) \
		-o $@ $(LDFLAGS) $(HALIDE_SYSTEM_LDFLAGS)

# One specific pipeline
$(BIN)/$(ID)/random_pipeline.a $(BIN)/$(ID)/features.mp: $(BIN)/random_pipeline.generator
	@mkdir -p $(@D)
	@HL_RANDOM_DROPOUT=$(HL_RANDOM_DROPOUT) \
	HL_SEED=$(HL_SEED) \
	HL_BEAM_SIZE=$(HL_BEAM_SIZE) \
	HL_DEBUG_CODEGEN=1 \
	HL_DEBUG=1 \
	HL_MACHINE_PARAMS=$(HL_MACHINE_PARAMS) \
	HL_JSON_DUMP=$(BIN)/$(ID)/features.mp \
	$^ -g random_pipeline -o $(BIN)/$(ID) -f random_pipeline \
		target=$(HL_TARGET)-no_runtime auto_schedule=$(AUTO_SCHEDULE) \
		seed=$(PIPELINE_SEED) max_stages=$(PIPELINE_STAGES) \
		2>&1 > /dev/null
		# 2> $(BIN)/$(ID)/stderr.txt > $(BIN)/$(ID)/stdout.txt

# Binary to benchmark one given pipeline
$(BIN)/$(ID)/benchmark: benchmark.cpp $(BIN)/$(ID)/random_pipeline.a $(BIN)/runtime.a
	@mkdir -p $(@D)
	@$(CXX) $(CXXFLAGS) -I$(BIN)/$(ID) -I$(JSON) -Wall -O3 $^ -o $@ $(LDFLAGS)

# Benchmarking result file
$(BIN)/$(ID)/timing.mp: $(BIN)/$(ID)/benchmark $(BIN)/$(ID)/features.mp
	@$< $@ 2>&1 > /dev/null

# Target to actually run the benchmark
bench: $(BIN)/$(ID)/timing.mp

clean:
	@echo Cleaning binaries
	@rm -rf $(BIN)

# .INTERMEDIATE: $(BIN)/$(ID)/random_pipeline.a $(BIN)/$(ID)/benchmark
