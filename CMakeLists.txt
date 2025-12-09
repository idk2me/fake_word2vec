cmake_minimum_required(VERSION 3.16)

# Project name and language
project(embeddings LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Collect all .cpp files inside src/
file(GLOB SOURCES "src/*.cpp")

# Create an executable from those files
add_executable(embeddings ${SOURCES})
file(COPY data DESTINATION ${CMAKE_BINARY_DIR})

# Add include directory for headers
target_include_directories(embeddings PRIVATE include)
