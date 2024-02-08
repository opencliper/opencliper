cmake_minimum_required(VERSION 3.0)
project(OpenCLIPER)
include(ExternalProject)
include(GNUInstallDirs)
include(CheckTypeSize)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake_modules)

# Show all cmake variables in stdout
set (SHOW_CMAKE_VARIABLES OFF)
#set (SHOW_CMAKE_VARIABLES ON)

# RPATH settings: use full RPATH when at the build tree, relink with suitable RPATH when installing
# Taken from https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

# Default values for build variables
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type: None, Debug, Release, RelWithDebInfo, MinSizeRel, Install" FORCE)
endif()
message("Build type is: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_INSTALL
    "-O3 -DNDEBUG"
    CACHE STRING "Flags used by the C++ compiler during install builds."
    FORCE )
set(CMAKE_C_FLAGS_INSTALL
    "-O3 -DNDEBUG"
    CACHE STRING "Flags used by the C compiler during install builds."
    FORCE )
set(CMAKE_EXE_LINKER_FLAGS_INSTALL
    ""
    CACHE STRING "Flags used for linking binaries during install builds."
    FORCE )
set(CMAKE_SHARED_LINKER_FLAGS_INSTALL
    ""
    CACHE STRING "Flags used by the shared libraries linker during install builds."
    FORCE )
mark_as_advanced(
    CMAKE_CXX_FLAGS_INSTALL
    CMAKE_C_FLAGS_INSTALL
    CMAKE_EXE_LINKER_FLAGS_INSTALL
    CMAKE_SHARED_LINKER_FLAGS_INSTALL )

set(KERNEL_INCLUDE_DIR_SUFFIX "include" CACHE STRING "Suffix of include directory for OpenCL kernels")
set(KERNEL_SOURCE_DIR_SUFFIX "share/OpenCLIPER/kernels" CACHE STRING "Suffix of source directory for OpenCL kernels")
set(KERNEL_USER_DIR ".OpenCLIPER/kernels" CACHE STRING "User directory for kernel files (relative to $HOME")
set(DATA_DIR_SUFFIX "share/OpenCLIPER/data" CACHE STRING "Suffix of data directory for examples and tests")
set(DEBUG_OUTPUT_DIR_SUFFIX "output" CACHE STRING "Suffix of debug output directory for examples and tests")

# Any cleaner way to do this?
string(COMPARE EQUAL ${CMAKE_BUILD_TYPE} "Debug" BUILD_TYPE_IS_DEBUG)
string(COMPARE EQUAL ${CMAKE_BUILD_TYPE} "Release" BUILD_TYPE_IS_RELEASE)
string(COMPARE EQUAL ${CMAKE_BUILD_TYPE} "Install" BUILD_TYPE_IS_INSTALL)

if(BUILD_TYPE_IS_DEBUG OR BUILD_TYPE_IS_RELEASE)
    set(KERNEL_INCLUDE_DIR "${PROJECT_BINARY_DIR}/${KERNEL_INCLUDE_DIR_SUFFIX}")
    set(KERNEL_SOURCE_DIR "${PROJECT_BINARY_DIR}/${KERNEL_SOURCE_DIR_SUFFIX}")
    set(DATA_DIR "${PROJECT_SOURCE_DIR}/${DATA_DIR_SUFFIX}")
    set(DEBUG_OUTPUT_DIR "${PROJECT_BINARY_DIR}/${DEBUG_OUTPUT_DIR_SUFFIX}") # not used in Release builds
elseif(BUILD_TYPE_IS_INSTALL)
    set(KERNEL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/${KERNEL_INCLUDE_DIR_SUFFIX}")
    set(KERNEL_SOURCE_DIR "${CMAKE_INSTALL_PREFIX}/${KERNEL_SOURCE_DIR_SUFFIX}")
    set(DATA_DIR "${CMAKE_INSTALL_PREFIX}/${DATA_DIR_SUFFIX}")
    add_definitions(-DNDEBUG)
else()
    message(FATAL_ERROR "Unknown build type. Must be Debug, Release or Install")
endif(BUILD_TYPE_IS_DEBUG OR BUILD_TYPE_IS_RELEASE)

set(BUILD_TESTS ON CACHE BOOL "Build tests")
set(BUILD_CUDA_TESTS OFF CACHE BOOL "Build CUDA tests")

# Default values for using backtrace-cpp library for java-like statck trace cmake_host_system_information
#set (USE_BACKWARD_STACKTRACE OFF)
set (USE_BACKWARD_STACKTRACE ON)

#set(${CMAKE_VERBOSE_MAKEFILE} 1)

# Search dependencies
find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED core highgui)
find_package(CUDA)

find_package(HIP)
if(HIP_ROOT_DIR)
    find_path(HIP_INCLUDE_DIRS "hip/hip_runtime_api.h" HINTS "${HIP_ROOT_DIR}/include")
    if(HIP_INCLUDE_DIRS)
        message("-- Found HIP include directories: " ${HIP_INCLUDE_DIRS})
    else()
        message("-- Could NOT find HIP include directories")
    endif()

    find_library(HIP_LIBRARIES "hip_hcc" HINTS "${HIP_ROOT_DIR}/lib")
    if(HIP_LIBRARIES)
        message("-- Found HIP libraries: " ${HIP_LIBRARIES})
    else()
        message("-- Could NOT find HIP libraries")
    endif()

    if(HIP_INCLUDE_DIRS AND HIP_LIBRARIES)
        set(HAVE_HIP 1)
        set(HIP_HIPCC_FLAGS_DEBUG "-g" CACHE STRING "HIPCC Debug flags" FORCE)
        set(HIP_HIPCC_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "HIPCC Release flags" FORCE)

        if(BUILD_TYPE_IS_DEBUG)
            set(HIP_HIPCC_FLAGS ${HIP_HIPCC_FLAGS_DEBUG})
        else()
            set(HIP_HIPCC_FLAGS ${HIP_HIPCC_FLAGS_RELEASE})
        endif()

        set(CL2HIP_OBJECT "${PROJECT_BINARY_DIR}/hip/cl2hip.o")
        add_custom_command(OUTPUT ${CL2HIP_OBJECT}
                   COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_BINARY_DIR}/hip
                   COMMAND ${HIP_HIPCC_EXECUTABLE} ${HIP_HIPCC_FLAGS} -DHAVE_HIP -c -fPIC -I ${PROJECT_SOURCE_DIR}/include -o ${PROJECT_BINARY_DIR}/hip/cl2hip.o ${PROJECT_SOURCE_DIR}/hip/cl2hip.cpp
                   MAIN_DEPENDENCY ${PROJECT_SOURCE_DIR}/hip/cl2hip.cpp
                   DEPENDS ${PROJECT_SOURCE_DIR}/include/OpenCLIPER/cl2hip.hpp)

        find_path(ROCFFT_INCLUDE_DIRS "rocfft.h" HINTS "${HIP_INCLUDE_DIRS}")
        if(ROCFFT_INCLUDE_DIRS)
            message("-- Found rocFFT include directories: " ${ROCFFT_INCLUDE_DIRS})
        else()
            message("-- Could NOT find rocFFT include directories")
        endif()

        find_library(ROCFFT_LIBRARIES "rocfft" HINTS "${HIP_ROOT_DIR}/lib")
        if(ROCFFT_LIBRARIES)
            message("-- Found rocFFT libraries: " ${ROCFFT_LIBRARIES})
        else()
            message("-- Could NOT find rocFFT libraries")
        endif()

        if(ROCFFT_INCLUDE_DIRS AND ROCFFT_LIBRARIES)
            set(HAVE_ROCFFT 1)
        else()
            unset(ROCFFT_INCLUDE_DIRS CACHE)
            unset(ROCFFT_LIBRARIES CACHE)
        endif()
    else()
        unset(HIP_INCLUDE_DIRS CACHE)
        unset(HIP_LIBRARIES CACHE)
    endif()
endif()

# Modern OpenCL versions include the opencl.hpp file, which should be used instead of
# the traditional cl2.hpp if exists
if(EXISTS ${OpenCL_INCLUDE_DIRS}/CL/opencl.hpp)
    set(HAVE_OPENCL_HPP 1)
endif()

# Check if cl_device_topology_amd is defined in cl_ext.h
# and set HAVE_CL_DEVICE_TOPOLOGY_AMD accordingly to define it ourselves if needed
list(APPEND CMAKE_REQUIRED_INCLUDES ${OpenCL_INCLUDE_DIRS})
list(APPEND CMAKE_EXTRA_INCLUDE_FILES "CL/cl_ext.h")
CHECK_TYPE_SIZE(cl_device_topology_amd CL_DEVICE_TOPOLOGY_AMD_SIZE)
string(COMPARE NOTEQUAL "${CL_DEVICE_TOPOLOGY_AMD_SIZE}" "" HAVE_CL_DEVICE_TOPOLOGY_AMD)

find_library(MATIO_LIBRARY "matio")
if(MATIO_LIBRARY)
    message("-- Found matio library: " ${MATIO_LIBRARY})
else()
    message("-- Could NOT find libmatio")
endif()

find_path(MATIO_INCLUDE_DIRS "matio.h")
if(MATIO_INCLUDE_DIRS)
    message("-- Found matio include directories: " ${MATIO_INCLUDE_DIRS})
else()
    message("-- Could NOT find libmatio include directories")
endif()

#find_package(clFFT REQUIRED)
#if(clFFT_FOUND)
#    message("-- Found clFFT package: " -l${CLFFT_LIBRARIES} "; " ${CLFFT_INCLUDE_DIRS})
#else()
#    message("-- Could NOT find clFFT package")
#endif()

find_program(PATCH NAMES "patch")
if(PATCH)
    message("    patch... FOUND")
else()
    message("    patch... NOT FOUND")
endif()

# The CLBlast package
set(CLBLAST_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/contrib)
set(CLBLAST_LIB ${CLBLAST_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libclblast.so)
ExternalProject_Add(CLBlast GIT_REPOSITORY https://github.com/CNugteren/CLBlast.git
			    GIT_TAG 1.5.3
			    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CLBLAST_INSTALL_PREFIX}
			    BUILD_BYPRODUCTS ${CLBLAST_LIB})

# The clFFT package
# We need a patch to do FFTs at arbitrary offsets, so we can't use precompiled binaries
set(CLFFT_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/contrib)
set(CLFFT_LIB ${CLFFT_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libclFFT.so)
ExternalProject_Add(clFFT GIT_REPOSITORY https://github.com/clMathLibraries/clFFT.git
			    GIT_TAG v2.12.2
			    SOURCE_SUBDIR src
			    PATCH_COMMAND ${PATCH} -N -p1 -i ${PROJECT_SOURCE_DIR}/patches/clFFT_offsets.patch &&
				${PATCH} -N -p1 -i ${PROJECT_SOURCE_DIR}/patches/clFFT_OpenCL22.patch || true
			    CMAKE_ARGS -DCMAKE_CXX_FLAGS=-std=gnu++11\ -fpermissive\ -Wno-deprecated-declarations -DCMAKE_INSTALL_PREFIX=${CLFFT_INSTALL_PREFIX} -DSUFFIX_LIB=
			    BUILD_BYPRODUCTS ${CLFFT_LIB})

# The cvPlot package
# cvPlot does an automatic normalization of images before showing them, which tampers with our own normalization. We need a patch to disable it
# cvPlot destroys previously existing windows whose name equals newly created windows. We need to keep old ones for successive calls to show()
# Note: tests and examples make use of the constant MINSIGSTKSZ, which, in recent glibc versions is no longer a constant. They won't compile, at least with tag v1.2.2
# https://stackoverflow.com/questions/71454588/minsigstksz-error-after-update-in-my-manjaro-linux
set(CVPLOT_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/contrib)
set(CVPLOT_LIB ${CLBLAST_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libcvplot.so)
ExternalProject_Add(cvPlot GIT_REPOSITORY https://github.com/Profactor/cv-plot.git
			    GIT_TAG v1.2.2
			    SOURCE_SUBDIR CvPlot
			    PATCH_COMMAND ${PATCH} -N -p1 -i ${PROJECT_SOURCE_DIR}/patches/cvPlot-disable_normalization.patch && 
			        ${PATCH} -N -p1 -i ${PROJECT_SOURCE_DIR}/patches/cvPlot-keep_old_windows.patch || true
			    CMAKE_ARGS -DCVPLOT_HEADER_ONLY=OFF -DCVPLOT_WITH_TESTS=OFF -DCVPLOT_WITH_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON
                                -DCMAKE_INSTALL_PREFIX=${CLBLAST_INSTALL_PREFIX}
			    BUILD_BYPRODUCTS ${CVPLOT_LIB})

# Download data files for tests and examples
message("Downloading data files for tests and examples. Please be patient...")
list(APPEND DATA_FILES "MRIdata.mat")
list(APPEND DATA_FILES "MRIdataSOS.mat")
list(APPEND DATA_FILES "heartVideo.mat")
list(APPEND DATA_FILES "data_160x160_coils1_frames16_single.mat")
list(APPEND DATA_FILES "data_160x160_coils23_frames20_single.mat")
list(APPEND DATA_FILES "phantom3D_5fresp_20fcard.mat")
list(APPEND DATA_FILES "MRIdata3D.mat")
list(APPEND DATA_FILES "phantomSA_2mm_BH_20phases_8coils_AF4_cube.mat")
list(APPEND DATA_FILES "halfscan/171116_slc0.mat")
list(APPEND DATA_FILES "halfscan/171116_slc1.mat")
list(APPEND DATA_FILES "halfscan/171116_slc2.mat")
list(APPEND DATA_FILES "halfscan/171116_slc3.mat")
list(APPEND DATA_FILES "halfscan/171116_slc4.mat")
list(APPEND DATA_FILES "halfscan/171116_slc5.mat")
list(APPEND DATA_FILES "halfscan/171116_slc6.mat")
list(APPEND DATA_FILES "halfscan/171116_slc7.mat")
list(APPEND DATA_FILES "halfscan/171116_slc8.mat")
list(APPEND DATA_FILES "halfscan/171116_slc9.mat")
list(APPEND DATA_FILES "halfscan/171116_slc10.mat")
list(APPEND DATA_FILES "halfscan/171116_slc11.mat")
list(APPEND DATA_FILES "halfscan/171116_slc12.mat")
list(APPEND DATA_FILES "halfscan/171116_slc13.mat")

foreach(FILE IN LISTS DATA_FILES)
    if (NOT EXISTS "${PROJECT_SOURCE_DIR}/data/${FILE}")
        message("-- Downloading ${FILE}...")
        file(DOWNLOAD "http://opencliper.lpi.tel.uva.es/data/${FILE}" "${PROJECT_SOURCE_DIR}/data/${FILE}")
    endif()
endforeach(FILE)

# Create output directory for test and example results
if (NOT EXISTS "${PROJECT_SOURCE_DIR}/output")
	message("Creating output directory...")
	file(MAKE_DIRECTORY "${PROJECT_SOURCE_DIR}/output")
endif()
# Enable all warnings
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# We need C++11
if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

if(USE_BACKWARD_STACKTRACE)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/backward)
    find_package(Backward)
    # Adding C/C++ definition of USE_BACKWARD_STACKTRACE
    add_definitions(-DUSE_BACKWARD_STACKTRACE=1)
endif()

# Disable warning "ignoring attributes on template argument" (caused by the OpenCL headers)
# that clutters compiler output. Maybe this can break things in some obscure architectures
# due to misaligned memory accesses but actually we can't do anything about it
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes")

# Disable warning "unknown pragma" caused by the backward include file, which has pragmas
# for clang that GCC complains about
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wl,-wrap,clEnqueueNDRangeKernel")

# Add support library
add_subdirectory(LPISupport)

# Set include directores
include_directories(
    include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${LPISupport_SOURCE_DIR}/include
    ${OpenCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${HIP_INCLUDE_DIRS}
    ${ROCFFT_INCLUDE_DIRS}
    ${CLFFT_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/contrib/include
    ${MATIO_INCLUDE_DIRS}
    if(USE_BACKWARD_STACKTRACE)
        ${BACKWARD_INCLUDE_DIR}
    endif()
)

# Create configuration header file from build-time options
configure_file(include/OpenCLIPER/buildconfig.hpp.in include/OpenCLIPER/buildconfig.hpp)

# Build the main OpenCLIPER target
file(GLOB_RECURSE SOURCEFILES "src/*")
add_library(OpenCLIPER SHARED ${SOURCEFILES} ${CL2HIP_OBJECT})
add_dependencies(OpenCLIPER LPISupport CLBlast clFFT)

if(USE_BACKWARD_STACKTRACE)
    target_compile_definitions(OpenCLIPER PUBLIC ${BACKWARD_DEFINITIONS})
endif()

set(OpenCLIPER_libraries
    ${LPISupport_BINARY_DIR}/libLPISupport.so
    ${OpenCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${HIP_LIBRARIES}
    ${ROCFFT_LIBRARIES}
    ${CLFFT_LIBRARIES}
    ${CMAKE_BINARY_DIR}/contrib/${CMAKE_INSTALL_LIBDIR}/libclblast.so
    ${CMAKE_BINARY_DIR}/contrib/${CMAKE_INSTALL_LIBDIR}/libclFFT.so
    ${CMAKE_BINARY_DIR}/contrib/${CMAKE_INSTALL_LIBDIR}/libCvPlot.so
    ${MATIO_LIBRARY}
)

if(BUILD_CUDA_TESTS)
    list(APPEND OpenCLIPER_libraries ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
endif()

target_link_libraries(OpenCLIPER ${OpenCLIPER_libraries})

# In debug and release builds, leave kernel sources, etc. available from build tree
if(BUILD_TYPE_IS_DEBUG OR BUILD_TYPE_IS_RELEASE)
    # (include dir is already created by configure_file)
    add_custom_command(TARGET OpenCLIPER POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/include/OpenCLIPER/kernels ${KERNEL_INCLUDE_DIR}/OpenCLIPER/kernels)

    get_filename_component(KERNEL_SOURCE_DIR_PARENT ${KERNEL_SOURCE_DIR} DIRECTORY)
    add_custom_command(TARGET OpenCLIPER POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ${KERNEL_SOURCE_DIR_PARENT})
    add_custom_command(TARGET OpenCLIPER POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/src/kernels ${KERNEL_SOURCE_DIR})

    get_filename_component(DATA_DIR_PARENT ${DATA_DIR} DIRECTORY)
    add_custom_command(TARGET OpenCLIPER POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ${DATA_DIR_PARENT})
    add_custom_command(TARGET OpenCLIPER POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/data ${DATA_DIR})

    if(BUILD_TYPE_IS_DEBUG)
        add_custom_command(TARGET OpenCLIPER POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ${DEBUG_OUTPUT_DIR})
    endif(BUILD_TYPE_IS_DEBUG)
endif(BUILD_TYPE_IS_DEBUG OR BUILD_TYPE_IS_RELEASE)

install(TARGETS OpenCLIPER RUNTIME DESTINATION bin LIBRARY DESTINATION lib)
install(DIRECTORY ${CMAKE_BINARY_DIR}/contrib/${CMAKE_INSTALL_LIBDIR}/ DESTINATION lib)		# Note the trailing slash!
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/OpenCLIPER DESTINATION include)
install(FILES ${PROJECT_BINARY_DIR}/include/OpenCLIPER/buildconfig.hpp DESTINATION include/OpenCLIPER)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/LPISupport/include/LPISupport DESTINATION include)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/kernels/ DESTINATION ${KERNEL_SOURCE_DIR})	# Note the trailing slash!
install(DIRECTORY ${PROJECT_SOURCE_DIR}/data/ DESTINATION ${DATA_DIR})			# Note the trailing slash!

if(BUILD_TESTS)
    add_subdirectory(tests)
    add_subdirectory(performanceTests)
    add_subdirectory(examples)
endif()

# # Show all cmake variables
if (SHOW_CMAKE_VARIABLES)
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
endif()
