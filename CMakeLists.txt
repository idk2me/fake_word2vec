cmake_minimum_required(VERSION 3.16)
project(embeddings LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- w2v library ----
add_library(w2v STATIC
    src/w2v/util.cpp
)

target_include_directories(w2v PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ---- executable ----
add_executable(embeddings
    app/main.cpp
)

target_link_libraries(embeddings PRIVATE w2v)

file(COPY data DESTINATION ${CMAKE_BINARY_DIR})
