# Accept extracted release archives as well as installed CMake packages.
set(EIGEN_INCLUDE_DIR "$ENV{EIGEN_INCLUDE_DIR}" CACHE PATH "Directory containing Eigen/Core")
if(NOT EIGEN_INCLUDE_DIR)
    find_package(Eigen3 5.0.1 CONFIG QUIET)
    if(TARGET Eigen3::Eigen)
        return()
    endif()
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.prefix)"
        OUTPUT_VARIABLE _python_prefix OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    find_path(_eigen_include NAMES Eigen/Core
        HINTS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/eigen" "${_python_prefix}/include/eigen3"
        PATHS /usr/local/include/eigen3 /usr/include/eigen3 /opt/homebrew/include/eigen3
        PATH_SUFFIXES eigen3
    )
    if(_eigen_include)
        set(EIGEN_INCLUDE_DIR "${_eigen_include}")
    endif()
endif()

if(NOT EXISTS "${EIGEN_INCLUDE_DIR}/Eigen/Core")
    message(FATAL_ERROR
        "Eigen >=5.0.1 headers not found. Set EIGEN_INCLUDE_DIR to the directory containing Eigen/Core."
    )
endif()
if(NOT EXISTS "${EIGEN_INCLUDE_DIR}/Eigen/Version")
    message(FATAL_ERROR "Eigen >=5.0.1 required; older headers found at ${EIGEN_INCLUDE_DIR}")
endif()
file(READ "${EIGEN_INCLUDE_DIR}/Eigen/Version" _eigen_version_header)
foreach(_part IN ITEMS MAJOR MINOR PATCH)
    string(REGEX MATCH "#define EIGEN_${_part}_VERSION[ \t]+([0-9]+)" _match "${_eigen_version_header}")
    if(NOT _match)
        message(FATAL_ERROR "Cannot read Eigen version from ${EIGEN_INCLUDE_DIR}/Eigen/Version")
    endif()
    set(_eigen_${_part} "${CMAKE_MATCH_1}")
endforeach()
set(_eigen_version "${_eigen_MAJOR}.${_eigen_MINOR}.${_eigen_PATCH}")
if(_eigen_version VERSION_LESS 5.0.1)
    message(FATAL_ERROR "Eigen >=5.0.1 required; found ${_eigen_version} at ${EIGEN_INCLUDE_DIR}")
endif()
message(STATUS "Found Eigen ${_eigen_version}: ${EIGEN_INCLUDE_DIR}")
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${EIGEN_INCLUDE_DIR}")
