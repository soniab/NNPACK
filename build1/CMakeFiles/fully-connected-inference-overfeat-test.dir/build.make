# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK/build1

# Include any dependencies generated for this target.
include CMakeFiles/fully-connected-inference-overfeat-test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/fully-connected-inference-overfeat-test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/fully-connected-inference-overfeat-test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fully-connected-inference-overfeat-test.dir/flags.make

CMakeFiles/fully-connected-inference-overfeat-test.dir/test/fully-connected-inference/overfeat-fast.cc.o: CMakeFiles/fully-connected-inference-overfeat-test.dir/flags.make
CMakeFiles/fully-connected-inference-overfeat-test.dir/test/fully-connected-inference/overfeat-fast.cc.o: ../test/fully-connected-inference/overfeat-fast.cc
CMakeFiles/fully-connected-inference-overfeat-test.dir/test/fully-connected-inference/overfeat-fast.cc.o: CMakeFiles/fully-connected-inference-overfeat-test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/soniar/naiveDarknetNNPACKARMSVE/NNPACK/build1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fully-connected-inference-overfeat-test.dir/test/fully-connected-inference/overfeat-fast.cc.o"
	/home/soniar/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/fully-connected-inference-overfeat-test.dir/test/fully-connected-inference/overfeat-fast.cc.o -MF CMakeFiles/fully-connected-inference-overfeat-test.dir/test/fully-connected-inference/overfeat-fast.cc.o.d -o CMakeFiles/fully-connected-inference-overfeat-test.dir/test/fully-connected-inference/overfeat-fast.cc.o -c /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK/test/fully-connected-inference/overfeat-fast.cc

CMakeFiles/fully-connected-inference-overfeat-test.dir/test/fully-connected-inference/overfeat-fast.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fully-connected-inference-overfeat-test.dir/test/fully-connected-inference/overfeat-fast.cc.i"
	/home/soniar/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK/test/fully-connected-inference/overfeat-fast.cc > CMakeFiles/fully-connected-inference-overfeat-test.dir/test/fully-connected-inference/overfeat-fast.cc.i

CMakeFiles/fully-connected-inference-overfeat-test.dir/test/fully-connected-inference/overfeat-fast.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fully-connected-inference-overfeat-test.dir/test/fully-connected-inference/overfeat-fast.cc.s"
	/home/soniar/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK/test/fully-connected-inference/overfeat-fast.cc -o CMakeFiles/fully-connected-inference-overfeat-test.dir/test/fully-connected-inference/overfeat-fast.cc.s

# Object files for target fully-connected-inference-overfeat-test
fully__connected__inference__overfeat__test_OBJECTS = \
"CMakeFiles/fully-connected-inference-overfeat-test.dir/test/fully-connected-inference/overfeat-fast.cc.o"

# External object files for target fully-connected-inference-overfeat-test
fully__connected__inference__overfeat__test_EXTERNAL_OBJECTS =

fully-connected-inference-overfeat-test: CMakeFiles/fully-connected-inference-overfeat-test.dir/test/fully-connected-inference/overfeat-fast.cc.o
fully-connected-inference-overfeat-test: CMakeFiles/fully-connected-inference-overfeat-test.dir/build.make
fully-connected-inference-overfeat-test: libnnpack.a
fully-connected-inference-overfeat-test: libnnpack_reference_layers.a
fully-connected-inference-overfeat-test: deps/googletest/googlemock/gtest/libgtest.a
fully-connected-inference-overfeat-test: deps/cpuinfo/libcpuinfo.a
fully-connected-inference-overfeat-test: deps/cpuinfo/deps/clog/libclog.a
fully-connected-inference-overfeat-test: deps/pthreadpool/libpthreadpool.a
fully-connected-inference-overfeat-test: CMakeFiles/fully-connected-inference-overfeat-test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/soniar/naiveDarknetNNPACKARMSVE/NNPACK/build1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable fully-connected-inference-overfeat-test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fully-connected-inference-overfeat-test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fully-connected-inference-overfeat-test.dir/build: fully-connected-inference-overfeat-test
.PHONY : CMakeFiles/fully-connected-inference-overfeat-test.dir/build

CMakeFiles/fully-connected-inference-overfeat-test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fully-connected-inference-overfeat-test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fully-connected-inference-overfeat-test.dir/clean

CMakeFiles/fully-connected-inference-overfeat-test.dir/depend:
	cd /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK/build1 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK/build1 /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK/build1 /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK/build1/CMakeFiles/fully-connected-inference-overfeat-test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fully-connected-inference-overfeat-test.dir/depend

