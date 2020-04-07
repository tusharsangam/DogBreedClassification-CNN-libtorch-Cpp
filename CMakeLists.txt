cmake_minimum_required(VERSION 3.11.3)

# set(CMAKE_CXX_STANDARD 17)
project(pytorchcnn)
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} "-std=c++17 -pthread")

find_package(OpenCV REQUIRED)
find_package( Boost 1.40 COMPONENTS filesystem REQUIRED )
find_package (Torch REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIR})
include_directories(${TORCH_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})


# Find all executables
file(GLOB project_SRCS src/*.cpp) #src/*.h
#set_property(TARGET pytorchcnn PROPERTY CXX_STANDARD 14)
# Add project executable
add_executable(pytorchcnn ${project_SRCS})
target_link_libraries(pytorchcnn ${OpenCV_LIBRARIES})
target_link_libraries(pytorchcnn Boost::filesystem)
target_link_libraries(pytorchcnn  ${TORCH_LIBRARIES})