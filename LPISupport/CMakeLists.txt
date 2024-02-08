cmake_minimum_required(VERSION 2.8)
project(LPISupport)
include(ExternalProject)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake_modules)

# Default values for build variables
#set(BUILD_TESTS ON)

#set(${CMAKE_VERBOSE_MAKEFILE} 1)

if (CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -std=gnu++11")
else ()
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11")
 endif()

file(GLOB_RECURSE SOURCEFILES "src/*")
add_library(LPISupport SHARED ${SOURCEFILES})

if(USE_BACKWARD_STACKTRACE)
    include_directories(include ${BACKWARD_INCLUDE_DIRS})
    target_compile_definitions(LPISupport PUBLIC ${BACKWARD_DEFINITIONS})
    target_link_libraries(LPISupport ${BACKWARD_LIBRARIES})
else()
    include_directories(include)
endif()

install(TARGETS LPISupport RUNTIME DESTINATION bin LIBRARY DESTINATION lib)

# # Show all cmake variables
if (SHOW_CMAKE_VARIABLES)
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
endif()
