add_library(MyOpenMP INTERFACE)

find_library(OpenMP_libomp_LIBRARY
    NAMES libomp libiomp5
    HINTS ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES}
)

message(STATUS "OpenMP libomp lib: ${OpenMP_libomp_LIBRARY}")