cmake_minimum_required(VERSION 2.8)

project(Stereo)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#Open-cv
find_package(OpenCV 3.0 REQUIRED)
if(OpenCV_INCLUDE_DIRS)
	include_directories(${OpenCV_INCLUDE_DIRECTORIES})
	link_directories(${Boost_LIBRARY_DIRS})
endif(OpenCV_INCLUDE_DIRS)

include_directories(${PROJECT_SOURCE_DIR}/inc)
add_executable(Test ${PROJECT_SOURCE_DIR}/src/test.cpp
					${PROJECT_SOURCE_DIR}/src/stereo_functions.cpp
					${PROJECT_SOURCE_DIR}/inc/stereo_functions.hpp)
target_link_libraries(Test ${OpenCV_LIBS})
set_property(TARGET Test PROPERTY CXX_STANDARD 14)

include_directories(${PROJECT_SOURCE_DIR}/inc)
add_executable(Sample ${PROJECT_SOURCE_DIR}/src/sample.cpp)
target_link_libraries(Sample ${OpenCV_LIBS})
set_property(TARGET Sample PROPERTY CXX_STANDARD 14)
