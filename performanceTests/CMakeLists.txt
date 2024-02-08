if(BUILD_CUDA_TESTS)
    MESSAGE("CUDA tests building requested")
    MESSAGE("    CMAKE_MAJOR_VERSION: ${CMAKE_MAJOR_VERSION}, CMAKE_MINOR_VERSION: ${CMAKE_MINOR_VERSION}")
    if (NOT(CMAKE_VERSION VERSION_LESS "3.8"))
	MESSAGE ("    It is greater or equal than 3.8, CUDA compilation enabled")
	#cmake_minimum_required(VERSION 3.8)
	project(OpenCLIPER_peformanceTests LANGUAGES CXX CUDA)
    else (NOT(CMAKE_VERSION VERSION_LESS "3.8"))
	MESSAGE ("    It is less than 3.8, CUDA compilation disabled")
	set(BUILD_CUDA_TESTS OFF)
	project(OpenCLIPER_peformanceTests LANGUAGES CXX)
    endif (NOT(CMAKE_VERSION VERSION_LESS "3.8"))
else (BUILD_CUDA_TESTS)
    MESSAGE("CUDA tests building disabled at user's request")
endif (BUILD_CUDA_TESTS)

set(${CMAKE_VERBOSE_MAKEFILE} 1)
#set (CUDACXX nvcc)
#set (CMAKE_CUDA_COMPILER "nvcc")
#set (CMAKE_CXX_COMPILER "nvcc")

# --------------------------------------------------------------
set (EXTRAFLAGS "-I.")
if (CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -std=gnu++11")
else ()
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -Minfo -mp=allcores")
 endif()
# Adding C/C++ label definition with value of cmake variable PROJECT_SOURCE_DIR
add_definitions(-DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

add_executable(arrayMultSerial arrayMult.cpp PerformanceTestArrayOpParallel.cpp commonArrayMult.cpp vectorUtils.cpp)
target_link_libraries(arrayMultSerial LPISupport)

add_executable(arrayMultOpenMP_CPU arrayMult.cpp PerformanceTestArrayOpParallel.cpp commonArrayMult.cpp vectorUtils.cpp)
target_compile_definitions(arrayMultOpenMP_CPU PRIVATE USE_OPENMP_CPU=1)
target_link_libraries(arrayMultOpenMP_CPU LPISupport)

#add_executable(arrayMultOpenMP_GPU arrayMult.cpp commonArrayMult.cpp vectorUtils.cpp ../src/Utils.cpp ../src/SampleCollection.cpp)
#target_compile_definitions(arrayMultOpenMP_GPU PRIVATE USE_OPENMP=1 USE_GPU=1)
##target_link_libraries(arrayMultOpenMP_GPU OpenCLIPER)


#add_executable(arrayMultOpenACC_CPU arrayMult.cpp PerformanceTestArrayOpParallel.cpp commonArrayMult.cpp vectorUtils.cpp)
#target_compile_definitions(arrayMultOpenACC_CPU PRIVATE USE_OPENACC_CPU=1)
#target_link_libraries(arrayMultOpenACC_CPU LPISupport)

#set_target_properties(arrayMultOpenACC PROPERTIES COMPILE_FLAGS "-fopenacc -fopt-info-note-omp")
#set_target_properties(arrayMultOpenACC PROPERTIES LINK_FLAGS "-fopenacc -fopt-info-note-omp")
#if (CMAKE_COMPILER_IS_GNUCXX)
#  set_target_properties(arrayMultOpenACC PROPERTIES COMPILE_FLAGS "-fopenacc")
#  set_target_properties(arrayMultOpenACC PROPERTIES LINK_FLAGS "-fopenacc")
#else()
#  set_target_properties(arrayMultOpenACC PROPERTIES COMPILE_FLAGS "-acc")
#  set_target_properties(arrayMultOpenACC PROPERTIES LINK_FLAGS "-acc")
#endif()

add_executable(arrayAddSerial arrayAdd.cpp PerformanceTestArrayOpParallel.cpp commonArrayMult.cpp vectorUtils.cpp)
target_link_libraries(arrayAddSerial LPISupport)

add_executable(arrayAddOpenMP_CPU arrayAdd.cpp PerformanceTestArrayOpParallel.cpp commonArrayMult.cpp vectorUtils.cpp)
target_compile_definitions(arrayAddOpenMP_CPU PRIVATE USE_OPENMP_CPU=1)
target_link_libraries(arrayAddOpenMP_CPU LPISupport)

#add_executable(arrayAddOpenMP_GPU arrayAdd.cpp commonArrayMult.cpp vectorUtils.cpp ../src/Utils.cpp ../src/SampleCollection.cpp)
#target_compile_definitions(arrayAddOpenMP_GPU PRIVATE USE_OPENMP=1 USE_GPU=1)

#add_executable(arrayAddOpenACC_CPU arrayAdd.cpp PerformanceTestArrayOpParallel.cpp commonArrayMult.cpp vectorUtils.cpp)
#target_compile_definitions(arrayAddOpenACC_CPU PRIVATE USE_OPENACC_CPU=1)
#target_link_libraries(arrayAddOpenACC_CPU LPISupport)

add_executable(arrayMultOpenCLIPER arrayMultOpenCLIPER.cpp PerformanceTestArrayOpenCLIPER.cpp commonArrayMult.cpp vectorUtils.cpp)
add_executable(arrayAddOpenCLIPER arrayAddOpenCLIPER.cpp PerformanceTestArrayOpenCLIPER.cpp commonArrayMult.cpp vectorUtils.cpp)

if (USE_BACKWARD_STACKTRACE)
    include_directories(include ${BACKWARD_INCLUDE_DIRS})
    target_compile_definitions(arrayAddOpenCLIPER PUBLIC ${BACKWARD_DEFINITIONS})
    target_compile_definitions(arrayMultOpenCLIPER PUBLIC ${BACKWARD_DEFINITIONS} )
    target_link_libraries(arrayAddOpenCLIPER OpenCLIPER ${BACKWARD_LIBRARIES})
    target_link_libraries(arrayMultOpenCLIPER OpenCLIPER ${BACKWARD_LIBRARIES})
else (USE_BACKWARD_STACKTRACE)
    include_directories(include)
    target_link_libraries(arrayAddOpenCLIPER OpenCLIPER)
    target_link_libraries(arrayMultOpenCLIPER OpenCLIPER)
endif (USE_BACKWARD_STACKTRACE)

if (CMAKE_COMPILER_IS_GNUCXX)
  #set_target_properties(arrayMultOpenMP_CPU arrayMultOpenMP_GPU PROPERTIES COMPILE_FLAGS -fopenmp)
  #set_target_properties(arrayMultOpenMP_CPU arrayMultOpenMP_GPU PROPERTIES LINK_FLAGS -fopenmp)
  set_target_properties(arrayMultOpenMP_CPU arrayAddOpenMP_CPU PROPERTIES COMPILE_FLAGS -fopenmp)
  set_target_properties(arrayMultOpenMP_CPU arrayAddOpenMP_CPU PROPERTIES LINK_FLAGS -fopenmp)
  #set_target_properties(arrayMultOpenACC_CPU arrayAddOpenACC_CPU PROPERTIES COMPILE_FLAGS -fopenacc)
  #set_target_properties(arrayMultOpenACC_CPU arrayAddOpenACC_CPU PROPERTIES LINK_FLAGS -fopenacc)
else (CMAKE_COMPILER_IS_GNUCXX)
  #set_target_properties(arrayMultOpenMP_CPU arrayMultOpenMP_GPU PROPERTIES COMPILE_FLAGS -mp)
  #set_target_properties(arrayMultOpenMP_CPU arrayMultOpenMP_GPU PROPERTIES LINK_FLAGS -mp)
  set_target_properties(arrayMultOpenMP_CPU arrayAddOpenMP_CPU PROPERTIES COMPILE_FLAGS -mp)
  set_target_properties(arrayMultOpenMP_CPU arrayAddOpenMP_CPU PROPERTIES LINK_FLAGS -mp)
  #set_target_properties(arrayMultOpenACC_CPU arrayAddOpenACC_CPU PROPERTIES COMPILE_FLAGS "-acc –ta=multicore")
  #set_target_properties(arrayMultOpenACC_CPU arrayAddOpenACC_CPU PROPERTIES LINK_FLAGS "-acc –ta=multicore")
endif (CMAKE_COMPILER_IS_GNUCXX)

if (BUILD_CUDA_TESTS)
    add_executable(arrayMultCUDA ArrayMultCUDA.cpp arrayMultCUDA.cu PerformanceTestArrayOpParallel.cpp commonArrayMult.cpp vectorUtils.cpp)
    target_link_libraries(arrayMultCUDA LPISupport)
    add_executable(arrayAddCUDA ArrayAddCUDA.cpp arrayAddCUDA.cu PerformanceTestArrayOpParallel.cpp commonArrayMult.cpp vectorUtils.cpp)
    target_link_libraries(arrayAddCUDA LPISupport)
    target_compile_features(arrayMultCUDA PUBLIC cxx_std_11)
    target_compile_features(arrayAddCUDA PUBLIC cxx_std_11)
    install(TARGETS
        arrayMultSerial
        arrayMultOpenMP_CPU
        #arrayMultOpenMP_GPU
        #arrayMultOpenACC_CPU
        arrayMultOpenCLIPER
        arrayMultCUDA
        arrayAddSerial
        arrayAddOpenMP_CPU
        #arrayAddOpenMP_GPU
        #arrayAddOpenACC_CPU
        arrayAddOpenCLIPER
        arrayAddCUDA
        RUNTIME DESTINATION bin)
else (BUILD_CUDA_TESTS)
    install(TARGETS
        arrayMultSerial
        arrayMultOpenMP_CPU
        #arrayMultOpenMP_GPU
        #arrayMultOpenACC_CPU
        arrayMultOpenCLIPER
        arrayAddSerial
        arrayAddOpenMP_CPU
        #arrayAddOpenMP_GPU
        #arrayAddtOpenACC_CPU
        arrayAddOpenCLIPER
        RUNTIME DESTINATION bin)
endif (BUILD_CUDA_TESTS)

#set (SHOW_CMAKE_VARIABLES on)
# # Show all cmake variables
if (SHOW_CMAKE_VARIABLES)
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
endif()
