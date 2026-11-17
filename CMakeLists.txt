cmake_minimum_required(VERSION 3.26)
project(sppy LANGUAGES CXX)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
include(cmake/FindSpPyEigen.cmake)

set(_openmp_default OFF)
if("$ENV{SPPY_OPENMP}" STREQUAL "1")
    set(_openmp_default ON)
endif()
option(SPPY_OPENMP "Enable OpenMP in parallel sparse kernels" ${_openmp_default})
if(SPPY_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# Expand private Cython templates in the build tree, leaving the checkout clean.
set(_generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(_templates
    "${CMAKE_CURRENT_SOURCE_DIR}/sppy/csarray_base.pyx"
    "${CMAKE_CURRENT_SOURCE_DIR}/sppy/csarray_base.pxd"
    "${CMAKE_CURRENT_SOURCE_DIR}/sppy/csarray1d_base.pyx"
    "${CMAKE_CURRENT_SOURCE_DIR}/sppy/csarray1d_base.pxd"
)
set(_expanded
    "${_generated}/sppy/csarray_sub.pyx"
    "${_generated}/sppy/csarray_sub.pxd"
    "${_generated}/sppy/csarray1d_sub.pyx"
    "${_generated}/sppy/csarray1d_sub.pxd"
)
add_custom_command(
    OUTPUT ${_expanded}
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/sppy/TemplateExpander.py"
            --output-dir "${_generated}/sppy"
    DEPENDS ${_templates} "${CMAKE_CURRENT_SOURCE_DIR}/sppy/TemplateExpander.py"
    COMMENT "Expanding sparse array bindings"
    VERBATIM
)

foreach(_module IN ITEMS csarray_sub csarray1d_sub)
    set(_cpp "${_generated}/sppy/${_module}.cpp")
    add_custom_command(
        OUTPUT "${_cpp}"
        COMMAND "${Python_EXECUTABLE}" -m cython --cplus -3 -X embedsignature=True
                --module-name "sppy.${_module}"
                -I "${_generated}" -I "${CMAKE_CURRENT_SOURCE_DIR}"
                --output-file "${_cpp}" "${_generated}/sppy/${_module}.pyx"
        DEPENDS ${_expanded}
        COMMENT "Compiling ${_module} with Cython"
        VERBATIM
    )
    python_add_library(${_module} MODULE WITH_SOABI "${_cpp}")
    target_compile_features(${_module} PRIVATE cxx_std_17)
    target_compile_definitions(${_module} PRIVATE NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
    target_include_directories(${_module} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/sppy")
    target_link_libraries(${_module} PRIVATE Python::NumPy Eigen3::Eigen)
    if(SPPY_OPENMP)
        target_link_libraries(${_module} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    install(TARGETS ${_module} LIBRARY DESTINATION sppy RUNTIME DESTINATION sppy)
endforeach()
