cmake_minimum_required(VERSION 3.10)
project(LSS_Solver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Inclure directement le dossier eigen
include_directories("${CMAKE_SOURCE_DIR}/eigen")

set(SOURCES
    main2.cpp
    MatrixGenerator.cpp
    JacobiSolver.cpp
    GaussSeidelSolver.cpp
    SORSolver.cpp
    ConvergenceAnalyzer.cpp
)

add_executable(lss_solver ${SOURCES})
target_include_directories(lss_solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
