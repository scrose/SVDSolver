# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/boutrous/Workspace/C++/csc586c_svd

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/boutrous/Workspace/C++/csc586c_svd/cmake-build-profiler

# Include any dependencies generated for this target.
include CMakeFiles/csc586c_svd_timin.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/csc586c_svd_timin.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/csc586c_svd_timin.dir/flags.make

CMakeFiles/csc586c_svd_timin.dir/tests.cpp.o: CMakeFiles/csc586c_svd_timin.dir/flags.make
CMakeFiles/csc586c_svd_timin.dir/tests.cpp.o: ../tests.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/boutrous/Workspace/C++/csc586c_svd/cmake-build-profiler/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/csc586c_svd_timin.dir/tests.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/csc586c_svd_timin.dir/tests.cpp.o -c /Users/boutrous/Workspace/C++/csc586c_svd/tests.cpp

CMakeFiles/csc586c_svd_timin.dir/tests.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/csc586c_svd_timin.dir/tests.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/boutrous/Workspace/C++/csc586c_svd/tests.cpp > CMakeFiles/csc586c_svd_timin.dir/tests.cpp.i

CMakeFiles/csc586c_svd_timin.dir/tests.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/csc586c_svd_timin.dir/tests.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/boutrous/Workspace/C++/csc586c_svd/tests.cpp -o CMakeFiles/csc586c_svd_timin.dir/tests.cpp.s

CMakeFiles/csc586c_svd_timin.dir/benchmarking.cpp.o: CMakeFiles/csc586c_svd_timin.dir/flags.make
CMakeFiles/csc586c_svd_timin.dir/benchmarking.cpp.o: ../benchmarking.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/boutrous/Workspace/C++/csc586c_svd/cmake-build-profiler/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/csc586c_svd_timin.dir/benchmarking.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/csc586c_svd_timin.dir/benchmarking.cpp.o -c /Users/boutrous/Workspace/C++/csc586c_svd/benchmarking.cpp

CMakeFiles/csc586c_svd_timin.dir/benchmarking.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/csc586c_svd_timin.dir/benchmarking.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/boutrous/Workspace/C++/csc586c_svd/benchmarking.cpp > CMakeFiles/csc586c_svd_timin.dir/benchmarking.cpp.i

CMakeFiles/csc586c_svd_timin.dir/benchmarking.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/csc586c_svd_timin.dir/benchmarking.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/boutrous/Workspace/C++/csc586c_svd/benchmarking.cpp -o CMakeFiles/csc586c_svd_timin.dir/benchmarking.cpp.s

# Object files for target csc586c_svd_timin
csc586c_svd_timin_OBJECTS = \
"CMakeFiles/csc586c_svd_timin.dir/tests.cpp.o" \
"CMakeFiles/csc586c_svd_timin.dir/benchmarking.cpp.o"

# External object files for target csc586c_svd_timin
csc586c_svd_timin_EXTERNAL_OBJECTS =

csc586c_svd_timin: CMakeFiles/csc586c_svd_timin.dir/tests.cpp.o
csc586c_svd_timin: CMakeFiles/csc586c_svd_timin.dir/benchmarking.cpp.o
csc586c_svd_timin: CMakeFiles/csc586c_svd_timin.dir/build.make
csc586c_svd_timin: CMakeFiles/csc586c_svd_timin.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/boutrous/Workspace/C++/csc586c_svd/cmake-build-profiler/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable csc586c_svd_timin"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/csc586c_svd_timin.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/csc586c_svd_timin.dir/build: csc586c_svd_timin

.PHONY : CMakeFiles/csc586c_svd_timin.dir/build

CMakeFiles/csc586c_svd_timin.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/csc586c_svd_timin.dir/cmake_clean.cmake
.PHONY : CMakeFiles/csc586c_svd_timin.dir/clean

CMakeFiles/csc586c_svd_timin.dir/depend:
	cd /Users/boutrous/Workspace/C++/csc586c_svd/cmake-build-profiler && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/boutrous/Workspace/C++/csc586c_svd /Users/boutrous/Workspace/C++/csc586c_svd /Users/boutrous/Workspace/C++/csc586c_svd/cmake-build-profiler /Users/boutrous/Workspace/C++/csc586c_svd/cmake-build-profiler /Users/boutrous/Workspace/C++/csc586c_svd/cmake-build-profiler/CMakeFiles/csc586c_svd_timin.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/csc586c_svd_timin.dir/depend

