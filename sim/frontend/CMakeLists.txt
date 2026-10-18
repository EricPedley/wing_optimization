cmake_minimum_required(VERSION 3.16)
project(wingsim CXX)

# paths
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(SIMITL_ROOT "${REPO_ROOT}/SimITL" CACHE PATH "SimITL fork root")
set(SIMITL_BUILD "${SIMITL_ROOT}/build/linux" CACHE PATH "SimITL build dir")
set(RAYLIB_ROOT "$ENV{HOME}/code/PufferLib/raylib-5.5_linux_amd64"
    CACHE PATH "raylib bundle root (include/ + lib/)")

find_package(SDL2 REQUIRED)

add_executable(wingsim main.cpp)
target_compile_features(wingsim PUBLIC cxx_std_17)

target_include_directories(wingsim PRIVATE
  "${SIMITL_ROOT}/src"
  "${SIMITL_ROOT}/extern"
  "${SIMITL_ROOT}/tools/simitl-wing"       # wingcfg.h
  "${SIMITL_ROOT}/tools/simitl-playback"   # json.h
  "${SIMITL_ROOT}/extern/fmt/include"  # fmt headers
  "${RAYLIB_ROOT}/include"
  ${SDL2_INCLUDE_DIRS}
)

# fmt include dir fallback (fetch vs vendored differ)
if(EXISTS "${SIMITL_ROOT}/extern/fmt/include")
  target_include_directories(wingsim PRIVATE "${SIMITL_ROOT}/extern/fmt/include")
endif()

target_link_directories(wingsim PRIVATE
  "${RAYLIB_ROOT}/lib"
)

# link the freshly built libSimITL.so directly (the installed copy can be stale)
target_link_libraries(wingsim PRIVATE
  "${SIMITL_BUILD}/src/libSimITL.so"
  raylib
  ${SDL2_LIBRARIES}
  m
  pthread
  dl
  rt
)

# so the .so is found without LD_LIBRARY_PATH
set_target_properties(wingsim PROPERTIES
  BUILD_RPATH "${SIMITL_BUILD}/src"
  INSTALL_RPATH "${SIMITL_BUILD}/src")

# render-transform checks (no window needed)
add_executable(test_wingmesh test_wingmesh.cpp)
target_compile_features(test_wingmesh PUBLIC cxx_std_17)
target_include_directories(test_wingmesh PRIVATE
  "${SIMITL_ROOT}/src"
  "${SIMITL_ROOT}/tools/simitl-playback"   # json.h
  "${RAYLIB_ROOT}/include"
)
target_link_directories(test_wingmesh PRIVATE "${RAYLIB_ROOT}/lib")
target_link_libraries(test_wingmesh PRIVATE raylib m pthread dl rt)
enable_testing()
add_test(NAME wingmesh COMMAND test_wingmesh "${CMAKE_CURRENT_SOURCE_DIR}/../as_built.json")
