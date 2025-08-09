# Compilation variables
CXX = g++
CXXFLAGS = -ansi -pedantic -Wall -std=c++17
CXXFLAGS_DEBUG = -ansi -pedantic -Wall -std=c++17 -g -DMAP
TARGET = AI
BUILD_DIR := build

SOURCES = main.cpp \
		  src/GameEngine/GameEngine.cpp src/GameEngine/InputHandler.cpp src/GameEngine/Observation.cpp src/GameEngine/PlayerSelector.cpp #src/GameEngine/StateAnalyzer.cpp
OBJECTS = $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(SOURCES))

all: clean $(TARGET)

debug: CXXFLAGS := $(CXXFLAGS_DEBUG)
debug: clean $(TARGET)

# Compile the target executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

# Compile each source file into an object file
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Ensure build directory exists
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Clean up everything
clean:
	rm -rf $(BUILD_DIR) $(TARGET)