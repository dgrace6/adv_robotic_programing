# Comments | Program by: 

#
cmake_minimum_required(VERSION "3.16")

#The project name
project(part-1-lecture-8)

#C++ Compiler Version
set(CMAKE_CXX_STANDARD 14)

#find package to include as part of your compiler
find_package( OpenCV REQUIRED )

#include directories with opencv headers
include_directories( ${OpenCV_INCLUDE_DIRS} )

#unnecessary to show the includes
message( ${OpenCV_INCLUDE_DIRS} )

#-o output option for makefiles
add_executable(HW4_Devin_Nikhil.o HW4_Devin_Nikhil.cpp)

#set target link libraries
target_link_libraries(HW4_Devin_Nikhil.o ${OpenCV_LIBS})


#sometimes there is a need for add_dependencies(), but not in this case.
