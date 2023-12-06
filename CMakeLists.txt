cmake_minimum_required(VERSION 3.1...3.27)
project(USING-MLPACK VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

add_executable(main main.cpp)
target_include_directories(main PUBLIC /opt/homebrew/include)
target_compile_options(main PUBLIC -Wno-deprecated-declarations)
target_link_directories(main PUBLIC /opt/homebrew/lib)
target_link_libraries(main PUBLIC armadillo)
