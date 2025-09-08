cmake_minimum_required(VERSION 3.15)
project(Iteratifs CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Trouver Python ---
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# --- Eigen (si installé via package manager ou téléchargé en local) ---
find_package(Eigen3 REQUIRED)

# --- Inclure matplotlib-cpp (assume matplotlibcpp.h est dans ton projet) ---
include_directories(${PROJECT_SOURCE_DIR})

# --- Exécutable ---
add_executable(iteratifs main.cpp)

# --- Inclure headers Python et NumPy ---
target_include_directories(iteratifs PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# --- Lien avec Python ---
target_link_libraries(iteratifs PRIVATE Python3::Python)
