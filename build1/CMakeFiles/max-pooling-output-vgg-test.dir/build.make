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
include CMakeFiles/max-pooling-output-vgg-test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/max-pooling-output-vgg-test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/max-pooling-output-vgg-test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/max-pooling-output-vgg-test.dir/flags.make

CMakeFiles/max-pooling-output-vgg-test.dir/test/max-pooling-output/vgg-a.cc.o: CMakeFiles/max-pooling-output-vgg-test.dir/flags.make
CMakeFiles/max-pooling-output-vgg-test.dir/test/max-pooling-output/vgg-a.cc.o: ../test/max-pooling-output/vgg-a.cc
CMakeFiles/max-pooling-output-vgg-test.dir/test/max-pooling-output/vgg-a.cc.o: CMakeFiles/max-pooling-output-vgg-test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/soniar/naiveDarknetNNPACKARMSVE/NNPACK/build1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/max-pooling-output-vgg-test.dir/test/max-pooling-output/vgg-a.cc.o"
	/home/soniar/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/max-pooling-output-vgg-test.dir/test/max-pooling-output/vgg-a.cc.o -MF CMakeFiles/max-pooling-output-vgg-test.dir/test/max-pooling-output/vgg-a.cc.o.d -o CMakeFiles/max-pooling-output-vgg-test.dir/test/max-pooling-output/vgg-a.cc.o -c /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK/test/max-pooling-output/vgg-a.cc

CMakeFiles/max-pooling-output-vgg-test.dir/test/max-pooling-output/vgg-a.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/max-pooling-output-vgg-test.dir/test/max-pooling-output/vgg-a.cc.i"
	/home/soniar/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK/test/max-pooling-output/vgg-a.cc > CMakeFiles/max-pooling-output-vgg-test.dir/test/max-pooling-output/vgg-a.cc.i

CMakeFiles/max-pooling-output-vgg-test.dir/test/max-pooling-output/vgg-a.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/max-pooling-output-vgg-test.dir/test/max-pooling-output/vgg-a.cc.s"
	/home/soniar/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK/test/max-pooling-output/vgg-a.cc -o CMakeFiles/max-pooling-output-vgg-test.dir/test/max-pooling-output/vgg-a.cc.s

# Object files for target max-pooling-output-vgg-test
max__pooling__output__vgg__test_OBJECTS = \
"CMakeFiles/max-pooling-output-vgg-test.dir/test/max-pooling-output/vgg-a.cc.o"

# External object files for target max-pooling-output-vgg-test
max__pooling__output__vgg__test_EXTERNAL_OBJECTS =

max-pooling-output-vgg-test: CMakeFiles/max-pooling-output-vgg-test.dir/test/max-pooling-output/vgg-a.cc.o
max-pooling-output-vgg-test: CMakeFiles/max-pooling-output-vgg-test.dir/build.make
max-pooling-output-vgg-test: libnnpack.a
max-pooling-output-vgg-test: libnnpack_reference_layers.a
max-pooling-output-vgg-test: deps/googletest/googlemock/gtest/libgtest.a
max-pooling-output-vgg-test: deps/cpuinfo/libcpuinfo.a
max-pooling-output-vgg-test: deps/cpuinfo/deps/clog/libclog.a
max-pooling-output-vgg-test: deps/pthreadpool/libpthreadpool.a
max-pooling-output-vgg-test: CMakeFiles/max-pooling-output-vgg-test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/soniar/naiveDarknetNNPACKARMSVE/NNPACK/build1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable max-pooling-output-vgg-test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/max-pooling-output-vgg-test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/max-pooling-output-vgg-test.dir/build: max-pooling-output-vgg-test
.PHONY : CMakeFiles/max-pooling-output-vgg-test.dir/build

CMakeFiles/max-pooling-output-vgg-test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/max-pooling-output-vgg-test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/max-pooling-output-vgg-test.dir/clean

CMakeFiles/max-pooling-output-vgg-test.dir/depend:
	cd /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK/build1 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK/build1 /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK/build1 /home/soniar/naiveDarknetNNPACKARMSVE/NNPACK/build1/CMakeFiles/max-pooling-output-vgg-test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/max-pooling-output-vgg-test.dir/depend

