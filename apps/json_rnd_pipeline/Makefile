include ../support/Makefile.inc

JSON=$(HALIDE_SRC_PATH)/vendors/include

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
LDFLAGS += $(ZMQ_LIBS)

all: build

build_shared: $(BIN)/runtime.a

build: $(BIN)/$(ID)/benchmark

# Runtime, shared across random pipelines
$(BIN)/runtime.a: $(BIN)/random_pipeline.generator
	@$^ -r runtime target=$(HL_TARGET) -o $(BIN)

# Common generator
$(BIN)/random_pipeline.generator: random_pipeline_generator.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	@$(CXX) $(CXXFLAGS) -fno-rtti $(filter-out %.h,$^) \
		-o $@ $(LDFLAGS) $(HALIDE_SYSTEM_LDFLAGS)

$(BIN)/$(ID)/random_pipeline.a: $(BIN)/random_pipeline.generator
	@mkdir -p $(@D)
	@HL_RANDOM_DROPOUT=$(HL_RANDOM_DROPOUT) \
	HL_SEED=$(HL_SEED) \
	HL_BEAM_SIZE=$(HL_BEAM_SIZE) \
	HL_DEBUG_CODEGEN=1 \
	HL_DEBUG=1 \
	HL_MACHINE_PARAMS=$(HL_MACHINE_PARAMS) \
	HL_JSON_DUMP=$(BIN)/$(ID)/features.msgpack \
	$^ -g random_pipeline -o $(BIN)/$(ID) -f random_pipeline \
		target=$(HL_TARGET)-no_runtime auto_schedule=$(AUTO_SCHEDULE) \
		seed=$(PIPELINE_SEED) max_stages=$(PIPELINE_STAGES) \
		2> $(BIN)/$(ID)/stderr.txt > $(BIN)/$(ID)/stdout.txt

$(BIN)/$(ID)/benchmark: benchmark.cpp $(BIN)/$(ID)/random_pipeline.a $(BIN)/runtime.a
	@mkdir -p $(@D)
	@$(CXX) $(CXXFLAGS) -I$(BIN)/$(ID) -I$(JSON) -Wall -O3 $^ -o $@ $(LDFLAGS)

$(BIN)/$(ID)/timing.json: $(BIN)/$(ID)/benchmark
	@$^ $@ 2>&1 > /dev/null

bench: $(BIN)/$(ID)/timing.json

clean:
	@echo Cleaning binaries
	@rm -rf $(BIN)

