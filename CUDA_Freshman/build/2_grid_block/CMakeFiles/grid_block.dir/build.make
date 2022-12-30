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
include 2_grid_block/CMakeFiles/grid_block.dir/depend.make

# Include the progress variables for this target.
include 2_grid_block/CMakeFiles/grid_block.dir/progress.make

# Include the compile flags for this target's objects.
include 2_grid_block/CMakeFiles/grid_block.dir/flags.make

2_grid_block/CMakeFiles/grid_block.dir/grid_block.cu.o: 2_grid_block/CMakeFiles/grid_block.dir/flags.make
2_grid_block/CMakeFiles/grid_block.dir/grid_block.cu.o: ../2_grid_block/grid_block.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yu/codings/cuda-work/CUDA_Freshman/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object 2_grid_block/CMakeFiles/grid_block.dir/grid_block.cu.o"
	cd /home/yu/codings/cuda-work/CUDA_Freshman/build/2_grid_block && /usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/yu/codings/cuda-work/CUDA_Freshman/2_grid_block/grid_block.cu -o CMakeFiles/grid_block.dir/grid_block.cu.o

2_grid_block/CMakeFiles/grid_block.dir/grid_block.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/grid_block.dir/grid_block.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

2_grid_block/CMakeFiles/grid_block.dir/grid_block.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/grid_block.dir/grid_block.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target grid_block
grid_block_OBJECTS = \
"CMakeFiles/grid_block.dir/grid_block.cu.o"

# External object files for target grid_block
grid_block_EXTERNAL_OBJECTS =

2_grid_block/grid_block: 2_grid_block/CMakeFiles/grid_block.dir/grid_block.cu.o
2_grid_block/grid_block: 2_grid_block/CMakeFiles/grid_block.dir/build.make
2_grid_block/grid_block: 2_grid_block/CMakeFiles/grid_block.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yu/codings/cuda-work/CUDA_Freshman/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable grid_block"
	cd /home/yu/codings/cuda-work/CUDA_Freshman/build/2_grid_block && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/grid_block.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
2_grid_block/CMakeFiles/grid_block.dir/build: 2_grid_block/grid_block

.PHONY : 2_grid_block/CMakeFiles/grid_block.dir/build

2_grid_block/CMakeFiles/grid_block.dir/clean:
	cd /home/yu/codings/cuda-work/CUDA_Freshman/build/2_grid_block && $(CMAKE_COMMAND) -P CMakeFiles/grid_block.dir/cmake_clean.cmake
.PHONY : 2_grid_block/CMakeFiles/grid_block.dir/clean

2_grid_block/CMakeFiles/grid_block.dir/depend:
	cd /home/yu/codings/cuda-work/CUDA_Freshman/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yu/codings/cuda-work/CUDA_Freshman /home/yu/codings/cuda-work/CUDA_Freshman/2_grid_block /home/yu/codings/cuda-work/CUDA_Freshman/build /home/yu/codings/cuda-work/CUDA_Freshman/build/2_grid_block /home/yu/codings/cuda-work/CUDA_Freshman/build/2_grid_block/CMakeFiles/grid_block.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : 2_grid_block/CMakeFiles/grid_block.dir/depend

