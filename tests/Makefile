# Compilation variables
TEST_FLAGS = -std=c++17 -O0 -g -Wall -Wextra -D_GLIBCXX_ASSERTIONS -fmax-errors=3
TEST_INCLUDES = -I. -Isrc -Isrc/GameEngine -Isrc/DQN -Isrc/Minimax
TEST_HEADERS = $(wildcard tests/*.h src/*.h src/*/*.h)
TEST_ENGINE = src/GameEngine/GameEngine.cpp src/GameEngine/PlayerSelector.cpp
TEST_ENCODING = src/GameEngine/Observation.cpp src/GameEngine/StateAnalyzer.cpp
TEST_NETWORKS = src/DQN/CNN.cpp src/DQN/FCNN.cpp
TEST_DQN = $(TEST_ENGINE) $(TEST_ENCODING) $(TEST_NETWORKS) src/DQN/DQN.cpp src/DQN/ReplayBuffer.cpp

.PHONY: test

test: build/tests/RunTests
	./build/tests/RunTests $(TEST)

build/tests:
	@mkdir -p $@

build/tests/RunTests: tests/RunTests.cpp | build/tests
	$(CXX) $(TEST_FLAGS) $< -o $@

build/tests/game-engine: tests/GameEngineTests.cpp $(TEST_ENGINE)
build/tests/input-handler: tests/InputHandlerTests.cpp src/GameEngine/InputHandler.cpp

build/tests/game-loop: TEST_FLAGS += -DSNAKE_EXECUTABLE=\"$(CURDIR)/AI\"
build/tests/game-loop: tests/GameLoopTests.cpp | AI

build/tests/observation: tests/ObservationTests.cpp $(TEST_ENGINE) src/GameEngine/Observation.cpp
build/tests/state-analyzer: tests/StateAnalyzerTests.cpp $(TEST_ENGINE) src/GameEngine/StateAnalyzer.cpp
build/tests/minimax: tests/MinimaxTests.cpp $(TEST_ENGINE) src/Agent/MinimaxAgent.cpp src/Minimax/Minimax.cpp src/Minimax/SpaceRiskAnalyzer.cpp
build/tests/cnn: tests/CNNTests.cpp src/DQN/CNN.cpp
build/tests/fcnn: tests/FCNNTests.cpp src/DQN/FCNN.cpp
build/tests/replay-buffer: tests/ReplayBufferTests.cpp src/DQN/ReplayBuffer.cpp
build/tests/dqn: tests/DQNTests.cpp $(TEST_DQN)
build/tests/persistence: tests/PersistenceTests.cpp $(TEST_DQN) src/DQN/ModelPersistence.cpp

build/tests/%: tests/TestMain.cpp $(TEST_HEADERS) | build/tests
	$(CXX) $(TEST_FLAGS) $(TEST_INCLUDES) $(filter %.cpp,$^) -o $@
