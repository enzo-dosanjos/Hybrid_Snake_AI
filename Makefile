# Compilation variables
CXX = g++
CXXFLAGS = -ansi -pedantic -Wall -std=c++17
TARGET = AI
BUIL_DIR := build

SOURCES = main.cpp CNN.cpp FCNN.cpp Minimax.cpp NNAI.cpp ReplayBuffer.cpp SpaceRiskAnalyzer.cpp Utils.cpp
OBJECTS = $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(SOURCES))

all: $(TARGET)

# Compile the target executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

# Compile each source file into an object file
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up everything
clean:
	rm -rf $(BUILD_DIR) $(TARGET)