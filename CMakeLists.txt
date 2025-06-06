cmake_minimum_required(VERSION 3.18)

project( AMReX-Hydro
    # DESCRIPTION "A software framework for massively parallel, block-structured adaptive mesh refinement (AMR) applications"
#    HOMEPAGE_URL "https://amrex-codes.github.io/amrex/"
    LANGUAGES C CXX
    )

message(STATUS "CMake version: ${CMAKE_VERSION}")

#
# Check if CMAKE_BUILD_TYPE is given. If not, use default
#
if ( NOT CMAKE_BUILD_TYPE )
   set(CMAKE_CONFIGURATION_TYPES "Release;Debug;MinSizeRel;RelWithDebInfo")
   set(CMAKE_BUILD_TYPE Release
       CACHE STRING
       "Choose the build type, e.g. Release, Debug, or RelWithDebInfo." FORCE)
else ()
   message(STATUS "Build type set by user to '${CMAKE_BUILD_TYPE}'.")
endif()

#
# Options
#
set(HYDRO_SPACEDIM 3 CACHE STRING "Dimension of AMReX build: <2,3>")
option( HYDRO_PROJECTIONS  "Enable projections" YES)
option( HYDRO_EB     "Enable Embedded-Boundary support" YES)
option( HYDRO_NO_EB  "Disable EB support even if AMReX has EB support" NO)
option( HYDRO_OMP    "Enable OpenMP" NO )
option( HYDRO_MPI    "Enable MPI"   YES )
option( HYDRO_FFT    "Enable FFT"   NO )


set(HYDRO_GPU_BACKEND_VALUES NONE SYCL CUDA HIP)
set(HYDRO_GPU_BACKEND NONE CACHE STRING "On-node, accelerated GPU backend: <NONE,SYCL,CUDA,HIP>")
set_property(CACHE HYDRO_GPU_BACKEND PROPERTY STRINGS ${HYDRO_GPU_BACKEND_VALUES})
if (NOT HYDRO_GPU_BACKEND IN_LIST HYDRO_GPU_BACKEND_VALUES)
   message(FATAL_ERROR "HYDRO_GPU_BACKEND=${HYDRO_GPU_BACKEND} is not allowed."
      " Must be one of ${HYDRO_GPU_BACKEND_VALUES}")
endif ()

if (HYDRO_GPU_BACKEND STREQUAL "CUDA")
   enable_language(CUDA)

   if (CMAKE_VERSION GREATER_EQUAL 3.17)
      #The CUDAToolkit was added with version 3.17
      find_package(CUDAToolkit REQUIRED)
   endif()
endif ()

if (HYDRO_GPU_BACKEND STREQUAL "HIP")
  find_package(hip REQUIRED)
  find_package(rocrand REQUIRED)
  find_package(rocprim REQUIRED)
  find_package(hiprand REQUIRED)
  find_package(rocsparse REQUIRED)
endif ()

#
# Find AMReX, check for required components
#
if (NOT TARGET AMReX::amrex)
   set(AMREX_REQUIRED_COMPONENTS ${HYDRO_SPACEDIM}D DOUBLE)
   if (HYDRO_MPI)
      list(APPEND AMREX_REQUIRED_COMPONENTS MPI)
   endif ()
   if (HYDRO_OMP)
      list(APPEND AMREX_REQUIRED_COMPONENTS OMP)
   endif ()
   if (NOT HYDRO_GPU_BACKEND STREQUAL "NONE")
      list(APPEND AMREX_REQUIRED_COMPONENTS ${HYDRO_GPU_BACKEND})
   endif ()
   if (HYDRO_EB)
      list(APPEND AMREX_REQUIRED_COMPONENTS EB)
   endif ()
   if (HYDRO_FFT)
      list(APPEND AMREX_REQUIRED_COMPONENTS FFT)
   endif ()
   find_package(AMReX CONFIG REQUIRED ${AMREX_REQUIRED_COMPONENTS} )
endif ()

#
# Enable CUDA if requested
#
if (HYDRO_GPU_BACKEND STREQUAL "CUDA")
   # CMake 3.18+: CMAKE_CUDA_ARCHITECTURES
   # https://cmake.org/cmake/help/latest/policy/CMP0104.html
   if(POLICY CMP0104)
      cmake_policy(SET CMP0104 OLD)
   endif()
   if (CMAKE_VERSION VERSION_LESS 3.20)
     include(AMReX_SetupCUDA)
   endif()
   include(AMReXTargetHelpers)
endif ()


#
# Define the object library to compile
#
add_library(amrex_hydro OBJECT)
if (BUILD_SHARED_LIBS)
  set_target_properties(amrex_hydro PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
target_link_libraries(amrex_hydro PUBLIC AMReX::amrex)
add_subdirectory(Utils)
add_subdirectory(MOL)
add_subdirectory(Godunov)
add_subdirectory(BDS)

if (HYDRO_PROJECTIONS)
   add_subdirectory(Projections)
endif ()

if (HYDRO_EB AND HYDRO_NO_EB)
    message(FATAL_ERROR "HYDRO_EB and HYDRO_NO_EB cannot be enabled at the same time.")
endif()
if (HYDRO_EB)
   add_subdirectory(EBMOL)
   add_subdirectory(EBGodunov)
endif ()
if (HYDRO_NO_EB)
  target_compile_definitions(amrex_hydro PUBLIC HYDRO_NO_EB)
endif()

if (HYDRO_GPU_BACKEND STREQUAL "CUDA")
   setup_target_for_cuda_compilation(amrex_hydro)
endif ()

if ( NOT CMAKE_CXX_FLAGS )
   target_link_libraries(amrex_hydro PUBLIC AMReX::Flags_CXX)
endif ()


# Installation rules
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Create non-object library for use as external target
add_library(amrex_hydro_api)
if (BUILD_SHARED_LIBS)
  set_target_properties(amrex_hydro_api PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
if (HYDRO_GPU_BACKEND STREQUAL "CUDA")
   setup_target_for_cuda_compilation(amrex_hydro_api)
endif ()
target_link_libraries(amrex_hydro_api PUBLIC amrex_hydro)
add_library(${PROJECT_NAME}::amrex_hydro_api ALIAS amrex_hydro)

# Collect all headers and make them installable with the target
get_target_property(AMREX_HYDRO_INCLUDES amrex_hydro SOURCES)
list(FILTER AMREX_HYDRO_INCLUDES INCLUDE REGEX "\\.H")
set_target_properties(
  amrex_hydro PROPERTIES PUBLIC_HEADER "${AMREX_HYDRO_INCLUDES}")

# Install AMReX-Hydro
install(
  TARGETS amrex_hydro_api amrex_hydro
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME       DESTINATION bin
  ARCHIVE       DESTINATION lib
  LIBRARY       DESTINATION lib
  INCLUDES      DESTINATION include
  PUBLIC_HEADER DESTINATION include
  )

# Make AMReX-Hydro discoverable using `find_package`
install(
  EXPORT ${PROJECT_NAME}Targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )
configure_package_config_file(
  Tools/CMake/${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )
