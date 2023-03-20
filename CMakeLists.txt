cmake_minimum_required(VERSION 3.10)
project(cdlt-dialect LANGUAGES CXX C)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

# project(cdlt-dialect LANGUAGES CXX C)

# set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

# find_package(MLIR REQUIRED CONFIG)

# message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
# message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
# set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
# set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
# list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
# include(TableGen)
# include(AddLLVM)
# include(AddMLIR)
# include(HandleLLVMOptions)

# include_directories(${LLVM_INCLUDE_DIRS})
# include_directories(${MLIR_INCLUDE_DIRS})
# # include_directories(${PROJECT_SOURCE_DIR}/include)
# # include_directories(${PROJECT_BINARY_DIR}/include)
# link_directories(${LLVM_BUILD_LIBRARY_DIR})
# add_definitions(${LLVM_DEFINITIONS})

# set(LLVM_LIT_ARGS "-sv" CACHE STRING "lit default options")
# list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

# add_subdirectory(include)

# add_custom_target(Cdlt)
# set_target_properties(Cdlt PROPERTIES FOLDER include/cdlt)
# add_dependencies(Cdlt cdltOps)


# cmake_minimum_required(VERSION 3.13.4)


set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
# add_subdirectory(lib)
if(MLIR_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Enabling Python API")
  add_subdirectory(python)
endif()
# add_subdirectory(test)
# add_subdirectory(standalone-opt)
# add_subdirectory(standalone-translate)

add_custom_target(Cdlt ALL)
set_target_properties(Cdlt PROPERTIES FOLDER include/cdlt)
add_dependencies(Cdlt cdltbase)
