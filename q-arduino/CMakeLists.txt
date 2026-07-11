cmake_minimum_required(VERSION 3.10)
project(EdgeAIServer)

# uWebSockets generally requires C++17 or C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find zlib (required by uWebSockets)
find_package(ZLIB REQUIRED)

# Add the main executable
add_executable(edge_server main.cpp)

# Assuming uWebSockets is installed system-wide or via a package manager 
# that provides include paths, and uSockets is built and available.
find_path(UWS_INCLUDE_DIRS "App.h")
find_path(USOCKETS_INCLUDE_DIRS "libusockets.h")
find_library(USOCKETS_LIBRARY usockets)

# Include directories
target_include_directories(edge_server PRIVATE 
    ${UWS_INCLUDE_DIRS}
    ${USOCKETS_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# Link OpenCV, uSockets, ZLIB, and Pthreads
target_link_libraries(edge_server PRIVATE 
    ${OpenCV_LIBS}
    ${USOCKETS_LIBRARY}
    ZLIB::ZLIB
    pthread
)
