# Compilation variables
CXX = g++
CXXFLAGS = -pedantic -Wall -std=c++17
CXXFLAGS_DEBUG = -pedantic -Wall -std=c++17 -g -DMAP
TARGET = AI
BUILD_DIR := build

MINIMAX_SOURCES = src/Agent/MinimaxAgent.cpp \
          src/GameEngine/GameEngine.cpp \
          src/GameEngine/InputHandler.cpp \
          src/GameEngine/PlayerSelector.cpp \
          src/Minimax/Minimax.cpp \
          src/Minimax/MinimaxParamsFile.cpp \
          src/Minimax/SpaceRiskAnalyzer.cpp
SHARED_SOURCES = $(MINIMAX_SOURCES) \
          src/Agent/DQNAgent.cpp \
          src/Agent/HybridAgent.cpp \
          src/DQN/CNN.cpp \
          src/DQN/DQN.cpp \
          src/DQN/FCNN.cpp \
          src/DQN/ModelPersistence.cpp \
          src/DQN/ReplayBuffer.cpp \
          src/GameEngine/Observation.cpp \
          src/GameEngine/StateAnalyzer.cpp
PLAY_SOURCES = main.cpp $(SHARED_SOURCES)
TRAIN_SOURCES = train.cpp src/TrainingLoop.cpp src/DQN/ResultValidation.cpp src/DQN/Reward.cpp $(SHARED_SOURCES)
OPTIMISE_SOURCES = optimise.cpp src/DQN/ResultValidation.cpp $(MINIMAX_SOURCES)
PLAY_OBJECTS = $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(PLAY_SOURCES))
TRAIN_OBJECTS = $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(TRAIN_SOURCES))
OPTIMISE_OBJECTS = $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(OPTIMISE_SOURCES))

.PHONY: all debug clean

all: $(TARGET) train optimise

debug:
	$(MAKE) clean
	$(MAKE) CXXFLAGS="$(CXXFLAGS_DEBUG)" $(TARGET)

# Compile the target executable
$(TARGET): $(PLAY_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

train: $(TRAIN_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

optimise: $(OPTIMISE_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile each source file into an object file
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

# Ensure build directory exists
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Clean up everything
clean:
	rm -rf $(BUILD_DIR) $(TARGET) train optimise

-include $(sort $(PLAY_OBJECTS:.o=.d) $(TRAIN_OBJECTS:.o=.d) $(OPTIMISE_OBJECTS:.o=.d))

include tests/Makefile
