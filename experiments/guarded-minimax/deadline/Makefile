include experiments/tactical-guard/Makefile

DEADLINE := experiments/guarded-minimax/deadline

build/experiments/DeadlinePlayer: $(DEADLINE)/Player.cpp $(GUARD)/ProofSearch.cpp $(MINIMAX_SOURCES) $(GUARD_HEADERS)
	@mkdir -p $(@D)
	$(CXX) -std=c++17 -O2 -DNDEBUG $(filter %.cpp,$^) -o $@
