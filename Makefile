# Compilation variables
CXX = g++
CXXFLAGS = -pedantic -Wall -std=c++17
CXXFLAGS_DEBUG = -pedantic -Wall -std=c++17 -g -DMAP
TARGET = AI
BUILD_DIR := build

SOURCES = main.cpp \
		  src/GameEngine/GameEngine.cpp src/GameEngine/InputHandler.cpp src/GameEngine/Observation.cpp src/GameEngine/PlayerSelector.cpp #src/GameEngine/StateAnalyzer.cpp
OBJECTS = $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(SOURCES))

.PHONY: all debug clean

all: $(TARGET)

debug:
	$(MAKE) clean
	$(MAKE) CXXFLAGS="$(CXXFLAGS_DEBUG)" $(TARGET)

# Compile the target executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

# Compile each source file into an object file
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

# Ensure build directory exists
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Clean up everything
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

-include $(OBJECTS:.o=.d)

include tests/Makefile
