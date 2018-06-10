cmake_minimum_required(VERSION 3.5.1)

set(CMAKE_CXX_FLAGS "-std=c++1z -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g3 -Og -pg")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -s -DNDEBUG -march=native")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/findpackage)
set(EP_PREFIX "${PROJECT_SOURCE_DIR}/external")


include(cmake/package.cmake)

include_directories(
        ${PROJECT_SOURCE_DIR}/src
)

add_subdirectory(src)


# add test
enable_testing()
add_subdirectory(tests)
