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
include 1_check_dimension/CMakeFiles/check_dimension.dir/depend.make

# Include the progress variables for this target.
include 1_check_dimension/CMakeFiles/check_dimension.dir/progress.make

# Include the compile flags for this target's objects.
include 1_check_dimension/CMakeFiles/check_dimension.dir/flags.make

1_check_dimension/CMakeFiles/check_dimension.dir/check_dimension.cu.o: 1_check_dimension/CMakeFiles/check_dimension.dir/flags.make
1_check_dimension/CMakeFiles/check_dimension.dir/check_dimension.cu.o: ../1_check_dimension/check_dimension.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yu/codings/cuda-work/CUDA_Freshman/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object 1_check_dimension/CMakeFiles/check_dimension.dir/check_dimension.cu.o"
	cd /home/yu/codings/cuda-work/CUDA_Freshman/build/1_check_dimension && /usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/yu/codings/cuda-work/CUDA_Freshman/1_check_dimension/check_dimension.cu -o CMakeFiles/check_dimension.dir/check_dimension.cu.o

1_check_dimension/CMakeFiles/check_dimension.dir/check_dimension.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/check_dimension.dir/check_dimension.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

1_check_dimension/CMakeFiles/check_dimension.dir/check_dimension.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/check_dimension.dir/check_dimension.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target check_dimension
check_dimension_OBJECTS = \
"CMakeFiles/check_dimension.dir/check_dimension.cu.o"

# External object files for target check_dimension
check_dimension_EXTERNAL_OBJECTS =

1_check_dimension/check_dimension: 1_check_dimension/CMakeFiles/check_dimension.dir/check_dimension.cu.o
1_check_dimension/check_dimension: 1_check_dimension/CMakeFiles/check_dimension.dir/build.make
1_check_dimension/check_dimension: 1_check_dimension/CMakeFiles/check_dimension.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yu/codings/cuda-work/CUDA_Freshman/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable check_dimension"
	cd /home/yu/codings/cuda-work/CUDA_Freshman/build/1_check_dimension && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/check_dimension.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
1_check_dimension/CMakeFiles/check_dimension.dir/build: 1_check_dimension/check_dimension

.PHONY : 1_check_dimension/CMakeFiles/check_dimension.dir/build

1_check_dimension/CMakeFiles/check_dimension.dir/clean:
	cd /home/yu/codings/cuda-work/CUDA_Freshman/build/1_check_dimension && $(CMAKE_COMMAND) -P CMakeFiles/check_dimension.dir/cmake_clean.cmake
.PHONY : 1_check_dimension/CMakeFiles/check_dimension.dir/clean

1_check_dimension/CMakeFiles/check_dimension.dir/depend:
	cd /home/yu/codings/cuda-work/CUDA_Freshman/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yu/codings/cuda-work/CUDA_Freshman /home/yu/codings/cuda-work/CUDA_Freshman/1_check_dimension /home/yu/codings/cuda-work/CUDA_Freshman/build /home/yu/codings/cuda-work/CUDA_Freshman/build/1_check_dimension /home/yu/codings/cuda-work/CUDA_Freshman/build/1_check_dimension/CMakeFiles/check_dimension.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : 1_check_dimension/CMakeFiles/check_dimension.dir/depend

