# Invoke from the project root to reuse the same sources as the C++ player.
include Makefile

GUARD := experiments/tactical-guard
GUARD_HEADERS := $(wildcard src/*/*.h src/*.h) $(GUARD)/ProofSearch.h

build/experiments/Evaluate: $(GUARD)/Evaluate.cpp $(GUARD)/ProofSearch.cpp $(SHARED_SOURCES) $(GUARD_HEADERS)
	@mkdir -p $(@D)
	$(CXX) -std=c++17 -O2 -DNDEBUG $(filter %.cpp,$^) -o $@
