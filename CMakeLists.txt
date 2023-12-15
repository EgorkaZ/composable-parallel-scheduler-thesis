cmake_minimum_required(VERSION 3.15)

project(composable-parallel-scheduler)

# --------------------- Initial setup --------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -Wno-unknown-attributes")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
message(STATUS "CMake module path: ${CMAKE_MODULE_PATH}")

# --------------------- Setup backends -------------------------
option(USE_OMP "Use OpenMP" OFF)
option(USE_LB4OMP "Use LB4OMP" OFF)
if (${USE_LB4OMP})
    set(USE_OMP ON)
endif()
option(USE_TBB "Use oneTBB" OFF)
option(USE_EIGEN "Use Eigen" OFF)

message(STATUS "Use OpenMP: ${USE_OMP}")
message(STATUS "Use LB4OMP: ${USE_LB4OMP}")
message(STATUS "Use oneTBB: ${USE_TBB}")
message(STATUS "Use Eigen:  ${USE_EIGEN}")

set(MODES)

#- ---------------------- OpenMP/LB4OMP -------------------------
set(OMP_MODES OMP_STATIC OMP_DYNAMIC_NONMONOTONIC OMP_GUIDED_NONMONOTONIC)
if (${USE_LB4OMP})
    message(STATUS "------------------------- LB4OMP dependency -------------------------")
    set(OPENMP_STANDALONE_BUILD TRUE)
    set(LIBOMP_HAVE___RDTSC TRUE)
    set(LIBOMP_HAVE_X86INTRIN_H TRUE)

    include(FetchContent)
    FetchContent_Declare(
        lb4omp
        GIT_REPOSITORY https://github.com/lejabque/LB4OMP.git
    )
    FetchContent_MakeAvailable(lb4omp)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

    # Setup OMP target for parlay
    set(OpenMP_CXX_FOUND TRUE)
    set(LIB_OMP_TARGET "omp")
    list(APPEND BACKEND_DEPS ${LIB_OMP_TARGET})

    list(APPEND MODES ${OMP_MODES} OMP_RUNTIME)
elseif (${USE_OMP})
    message(STATUS "------------------------- OpenMP dependency -------------------------")
    if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        message(FATAL_ERROR "It was too hard to make gcc link with llvm's libomp, so it's forbidden")
    endif()
    find_package(OpenMP REQUIRED)
    message(STATUS "OpenMP flags:     ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP libs:      ${OpenMP_CXX_LIBRARIES}")

    # Setup OMP target for parlay
    set(LIB_OMP_TARGET OpenMP::OpenMP_CXX)
    list(APPEND BACKEND_DEPS ${LIB_OMP_TARGET})

    list(APPEND MODES ${OMP_MODES})
endif()

if (${USE_TBB})
    message(STATUS "------------------------- oneTBB dependency -------------------------")
    # if (NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    #     message(FATAL_ERROR "OneTBB breaks on ARM if not built with gcc")
    # endif()

    include(FetchContent)
    FetchContent_Declare(
        onetbb
        GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
        GIT_TAG v2021.11.0
    )
    option(TBB_TEST OFF)
    set(TBB_FOUND TRUE)
    FetchContent_MakeAvailable(onetbb)

    list(APPEND MODES TBB_SIMPLE TBB_AUTO TBB_AFFINITY TBB_CONST_AFFINITY)
endif()

if (${USE_EIGEN})
    message(STATUS "------------------------- Eigen dependency -------------------------")
    list(APPEND MODES EIGEN_SIMPLE EIGEN_TIMESPAN EIGEN_TIMESPAN_GRAINSIZE EIGEN_STATIC)
endif()

# --------------------- Setup parlay ----------------------------
set(PARLAY_OPENMP ${USE_OMP})
set(PARLAY_TBB ${USE_TBB})
set(PARLAY_EIGEN ${USE_EIGEN})
add_subdirectory(parlaylib)

# --------------- Setup definition by backend -------------------

function(add_target target exec mode)
    add_executable(${target} ${exec})
    target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/include")
    target_link_libraries(${target} PUBLIC parlay)
    if (mode MATCHES "^SERIAL")
      target_compile_definitions(${target} PRIVATE SERIAL=1)
    elseif (mode MATCHES "^OMP")
      target_compile_definitions(${target} PRIVATE OMP_MODE=${mode})
    elseif (mode MATCHES "^TBB")
      target_compile_definitions(${target} PRIVATE TBB_MODE=${mode})
    elseif (mode MATCHES "^EIGEN")
      target_compile_definitions(${target} PRIVATE EIGEN_MODE=${mode})
    endif()
endfunction()

add_subdirectory(benchmarks)
