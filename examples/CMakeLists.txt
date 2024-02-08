cmake_minimum_required(VERSION 2.8)
project(OpenCLIPER_examples)
set(${CMAKE_VERBOSE_MAKEFILE} 1)

# --------------------------------------------------------------
set (EXTRAFLAGS "")
if (CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++11")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

# adding dependencies for executable
if (USE_BACKWARD_STACKTRACE)
    link_libraries(${BACKWARD_LIBRARIES} OpenCLIPER)
    add_executable(simpleExample simpleExample.cpp ../performanceTests/PerformanceTestArrayOpParallel.cpp ${PROJECT_SOURCE_DIR}/../backward/backward.cpp)
    add_executable(simpleMRIRecon simpleMRIRecon.cpp ../performanceTests/PerformanceTestArrayOpParallel.cpp ${PROJECT_SOURCE_DIR}/../backward/backward.cpp)
    add_executable(simpleMRIReconSOS simpleMRIReconSOS.cpp ${PROJECT_SOURCE_DIR}/../backward/backward.cpp)

    if(HIP_INCLUDE_DIRS AND HIP_LIBRARIES AND ROCFFT_INCLUDE_DIRS AND ROCFFT_LIBRARIES)
        add_executable(hipExample hipExample.cpp ../performanceTests/PerformanceTestArrayOpParallel.cpp ${PROJECT_SOURCE_DIR}/../backward/backward.cpp)
    endif()
else()
    link_libraries(OpenCLIPER)
    add_executable(simpleExample simpleExample.cpp ../performanceTests/PerformanceTestArrayOpParallel.cpp)
    add_executable(simpleMRIRecon simpleMRIRecon.cpp ../performanceTests/PerformanceTestArrayOpParallel.cpp)
    add_executable(simpleMRIReconSOS simpleMRIReconSOS.cpp)

    if(HIP_INCLUDE_DIRS AND HIP_LIBRARIES AND ROCFFT_INCLUDE_DIRS AND ROCFFT_LIBRARIES)
        add_executable(hipExample hipExample.cpp ../performanceTests/PerformanceTestArrayOpParallel.cpp)
    endif()
endif()

install(TARGETS simpleExample
    		simpleMRIRecon
    		simpleMRIReconSOS
        RUNTIME DESTINATION bin)

# # Show all cmake variables
if (SHOW_CMAKE_VARIABLES)
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
endif()

