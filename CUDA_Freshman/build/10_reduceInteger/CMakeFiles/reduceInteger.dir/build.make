# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yu/codings/cuda-work/CUDA_Freshman

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yu/codings/cuda-work/CUDA_Freshman/build

# Include any dependencies generated for this target.
include 10_reduceInteger/CMakeFiles/reduceInteger.dir/depend.make

# Include the progress variables for this target.
include 10_reduceInteger/CMakeFiles/reduceInteger.dir/progress.make

# Include the compile flags for this target's objects.
include 10_reduceInteger/CMakeFiles/reduceInteger.dir/flags.make

10_reduceInteger/CMakeFiles/reduceInteger.dir/reduceInteger.cu.o: 10_reduceInteger/CMakeFiles/reduceInteger.dir/flags.make
10_reduceInteger/CMakeFiles/reduceInteger.dir/reduceInteger.cu.o: ../10_reduceInteger/reduceInteger.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yu/codings/cuda-work/CUDA_Freshman/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object 10_reduceInteger/CMakeFiles/reduceInteger.dir/reduceInteger.cu.o"
	cd /home/yu/codings/cuda-work/CUDA_Freshman/build/10_reduceInteger && /usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/yu/codings/cuda-work/CUDA_Freshman/10_reduceInteger/reduceInteger.cu -o CMakeFiles/reduceInteger.dir/reduceInteger.cu.o

10_reduceInteger/CMakeFiles/reduceInteger.dir/reduceInteger.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/reduceInteger.dir/reduceInteger.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

10_reduceInteger/CMakeFiles/reduceInteger.dir/reduceInteger.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/reduceInteger.dir/reduceInteger.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target reduceInteger
reduceInteger_OBJECTS = \
"CMakeFiles/reduceInteger.dir/reduceInteger.cu.o"

# External object files for target reduceInteger
reduceInteger_EXTERNAL_OBJECTS =

10_reduceInteger/reduceInteger: 10_reduceInteger/CMakeFiles/reduceInteger.dir/reduceInteger.cu.o
10_reduceInteger/reduceInteger: 10_reduceInteger/CMakeFiles/reduceInteger.dir/build.make
10_reduceInteger/reduceInteger: 10_reduceInteger/CMakeFiles/reduceInteger.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yu/codings/cuda-work/CUDA_Freshman/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable reduceInteger"
	cd /home/yu/codings/cuda-work/CUDA_Freshman/build/10_reduceInteger && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reduceInteger.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
10_reduceInteger/CMakeFiles/reduceInteger.dir/build: 10_reduceInteger/reduceInteger

.PHONY : 10_reduceInteger/CMakeFiles/reduceInteger.dir/build

10_reduceInteger/CMakeFiles/reduceInteger.dir/clean:
	cd /home/yu/codings/cuda-work/CUDA_Freshman/build/10_reduceInteger && $(CMAKE_COMMAND) -P CMakeFiles/reduceInteger.dir/cmake_clean.cmake
.PHONY : 10_reduceInteger/CMakeFiles/reduceInteger.dir/clean

10_reduceInteger/CMakeFiles/reduceInteger.dir/depend:
	cd /home/yu/codings/cuda-work/CUDA_Freshman/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yu/codings/cuda-work/CUDA_Freshman /home/yu/codings/cuda-work/CUDA_Freshman/10_reduceInteger /home/yu/codings/cuda-work/CUDA_Freshman/build /home/yu/codings/cuda-work/CUDA_Freshman/build/10_reduceInteger /home/yu/codings/cuda-work/CUDA_Freshman/build/10_reduceInteger/CMakeFiles/reduceInteger.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : 10_reduceInteger/CMakeFiles/reduceInteger.dir/depend

