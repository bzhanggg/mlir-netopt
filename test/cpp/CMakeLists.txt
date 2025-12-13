cmake_minimum_required(VERSION 3.10)

project(spmc.u.t LANGUAGES CXX)

# compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# source
add_executable(spmc.u.t spmc.cpp)

# link threads if needed
find_package(Threads REQUIRED)
target_link_libraries(spmc.u.t PRIVATE Threads::Threads)

# reasonable warnings
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(spmc.u.t PRIVATE -Wall -Wextra -Wpedantic)
endif()