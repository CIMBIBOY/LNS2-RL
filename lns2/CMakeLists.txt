cmake_minimum_required(VERSION 3.17)
project(my_lns2)

set(CMAKE_CXX_STANDARD 17)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "RELEASE")
ENDIF()

include_directories("inc")

find_package(Boost REQUIRED COMPONENTS program_options system filesystem)
include_directories(${Boost_INCLUDE_DIRS})

# OpenCV for optional image saving/debugging
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 🔁 Rerun (static or shared build expected)
# Replace this path with your actual rerun-cpp install path
include_directories(/usr/local/include/rerun)
link_directories(/usr/local/lib)

add_subdirectory(pybind11)
pybind11_add_module(my_lns2 main.cpp)
target_link_libraries(my_lns2 PRIVATE ${Boost_LIBRARIES} ${OpenCV_LIBS} rerun_cpp)

add_executable(mylns2_tester mylns2_tester.cpp
               Instance.cpp ConstraintTable.cpp PathTable.cpp ReservationTable.cpp
               SIPP.cpp SingleAgentSolver.cpp common.cpp mylns2.cpp)
target_link_libraries(mylns2_tester PRIVATE ${Boost_LIBRARIES} ${OpenCV_LIBS} rerun_cpp)