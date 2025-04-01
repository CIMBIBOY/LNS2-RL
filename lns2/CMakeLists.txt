cmake_minimum_required(VERSION 3.17)
project(my_lns2)

set(CMAKE_CXX_STANDARD 17)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "RELEASE")
ENDIF()

include_directories("inc")

find_package(Boost REQUIRED COMPONENTS program_options system filesystem)
include_directories( ${Boost_INCLUDE_DIRS} )

# Find and include OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find and include ROS packages
find_package(catkin REQUIRED COMPONENTS
    roscpp
    nav_msgs
    std_msgs
    geometry_msgs
    visualization_msgs
    cv_bridge
    sensor_msgs 
)

include_directories(
    ${catkin_INCLUDE_DIRS}
)

add_subdirectory(pybind11)
pybind11_add_module(my_lns2 main.cpp)
target_link_libraries(my_lns2 PRIVATE ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(mylns2_tester mylns2_tester.cpp
               Instance.cpp ConstraintTable.cpp PathTable.cpp ReservationTable.cpp
               SIPP.cpp SingleAgentSolver.cpp common.cpp mylns2.cpp)

target_link_libraries(mylns2_tester PRIVATE ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})
