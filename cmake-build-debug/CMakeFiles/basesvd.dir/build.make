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
CMAKE_SOURCE_DIR = /Users/boutrous/Workspace/C++/csc586c-project-final-report

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/boutrous/Workspace/C++/csc586c-project-final-report/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/basesvd.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/basesvd.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/basesvd.dir/flags.make

CMakeFiles/basesvd.dir/svd_multicore.cpp.o: CMakeFiles/basesvd.dir/flags.make
CMakeFiles/basesvd.dir/svd_multicore.cpp.o: ../svd_multicore.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/boutrous/Workspace/C++/csc586c-project-final-report/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/basesvd.dir/svd_multicore.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/basesvd.dir/svd_multicore.cpp.o -c /Users/boutrous/Workspace/C++/csc586c-project-final-report/svd_multicore.cpp

CMakeFiles/basesvd.dir/svd_multicore.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/basesvd.dir/svd_multicore.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/boutrous/Workspace/C++/csc586c-project-final-report/svd_multicore.cpp > CMakeFiles/basesvd.dir/svd_multicore.cpp.i

CMakeFiles/basesvd.dir/svd_multicore.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/basesvd.dir/svd_multicore.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/boutrous/Workspace/C++/csc586c-project-final-report/svd_multicore.cpp -o CMakeFiles/basesvd.dir/svd_multicore.cpp.s

# Object files for target basesvd
basesvd_OBJECTS = \
"CMakeFiles/basesvd.dir/svd_multicore.cpp.o"

# External object files for target basesvd
basesvd_EXTERNAL_OBJECTS =

basesvd: CMakeFiles/basesvd.dir/svd_multicore.cpp.o
basesvd: CMakeFiles/basesvd.dir/build.make
basesvd: CMakeFiles/basesvd.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/boutrous/Workspace/C++/csc586c-project-final-report/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable basesvd"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/basesvd.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/basesvd.dir/build: basesvd

.PHONY : CMakeFiles/basesvd.dir/build

CMakeFiles/basesvd.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/basesvd.dir/cmake_clean.cmake
.PHONY : CMakeFiles/basesvd.dir/clean

CMakeFiles/basesvd.dir/depend:
	cd /Users/boutrous/Workspace/C++/csc586c-project-final-report/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/boutrous/Workspace/C++/csc586c-project-final-report /Users/boutrous/Workspace/C++/csc586c-project-final-report /Users/boutrous/Workspace/C++/csc586c-project-final-report/cmake-build-debug /Users/boutrous/Workspace/C++/csc586c-project-final-report/cmake-build-debug /Users/boutrous/Workspace/C++/csc586c-project-final-report/cmake-build-debug/CMakeFiles/basesvd.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/basesvd.dir/depend
