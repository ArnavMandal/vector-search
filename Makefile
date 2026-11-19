CXX ?= c++
CXXFLAGS ?= -O3
BUILD ?= build
PYTHON ?= python3
PORT ?= 8765
PYBIND_INCLUDE ?= $(shell $(PYTHON) -c 'import pybind11; print(pybind11.get_include())')
PY_INCLUDE = $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_path("include"))')
EXT = $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
MODULE_FLAGS = $(if $(filter Darwin,$(shell uname -s)),-bundle -undefined dynamic_lookup,-shared)
# Some Apple Command Line Tools installs keep the standard headers only in the SDK.
ifeq ($(shell uname -s),Darwin)
ifeq ($(shell $(CXX) -dM -E -x c++ /dev/null | awk '/__apple_build_version__/ {print 1}'),1)
SDK_HEADERS := $(shell xcrun --show-sdk-path)/usr/include/c++/v1
ifneq ($(wildcard $(SDK_HEADERS)/cstddef),)
SDK_CPPFLAGS = -isystem "$(SDK_HEADERS)"
endif
endif
endif
FLAGS = $(CPPFLAGS) $(SDK_CPPFLAGS) $(CXXFLAGS) -std=c++20 -Wall -Wextra -Wpedantic -pthread -Iinclude
CORE = $(BUILD)/index.o $(BUILD)/storage.o $(BUILD)/distance.o $(BUILD)/hnsw.o

.PHONY: all bindings test test-python explorer
all: $(BUILD)/vector-search

$(BUILD):
	mkdir -p $@
$(BUILD)/%.o: src/%.cpp $(wildcard include/*.hpp src/*.hpp) | $(BUILD)
	$(CXX) $(FLAGS) -fPIC -c $< -o $@
$(BUILD)/libvector_search.a: $(CORE)
	$(AR) rcs $@ $^
$(BUILD)/vector-search: src/cli.cpp $(BUILD)/libvector_search.a
	$(CXX) $(FLAGS) $^ -o $@
ifneq ($(filter bindings test-python explorer,$(MAKECMDGOALS)),)
bindings: $(BUILD)/vector_search$(EXT)
$(BUILD)/vector_search$(EXT): python/bindings.cpp $(BUILD)/libvector_search.a
	$(CXX) $(FLAGS) -fPIC $(MODULE_FLAGS) -I"$(PY_INCLUDE)" -isystem "$(PYBIND_INCLUDE)" $^ -o $@
endif
$(BUILD)/test-engine: tests/engine.cpp $(BUILD)/libvector_search.a
	$(CXX) $(FLAGS) $^ -o $@
$(BUILD)/test-accelerators: tests/accelerators.cpp $(BUILD)/libvector_search.a
	$(CXX) $(FLAGS) $^ -o $@
test: $(BUILD)/test-engine $(BUILD)/test-accelerators $(BUILD)/vector-search
	$(BUILD)/test-accelerators
	$(BUILD)/test-engine
	sh tests/cli.sh "$(BUILD)/vector-search"
test-python: bindings
	PYTHONPATH="$(BUILD)" $(PYTHON) tests/test_python.py
explorer: bindings
	PYTHONPATH="$(BUILD)" $(PYTHON) explorer/server.py --port $(PORT)
