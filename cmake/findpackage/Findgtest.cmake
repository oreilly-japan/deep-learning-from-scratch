add_library(gtest UNKNOWN IMPORTED)
add_library(gmock UNKNOWN IMPORTED)

find_path(gtest_INCLUDE_DIR gtest/gtest.h)
mark_as_advanced(gtest_INCLUDE_DIR)

find_library(gtest_LIBRARY gtest)
mark_as_advanced(gtest_LIBRARY)

find_library(gtest_MAIN_LIBRARY gtest_main)
mark_as_advanced(gtest_MAIN_LIBRARY)

find_path(gmock_INCLUDE_DIR gmock/gmock.h)
mark_as_advanced(gmock_INCLUDE_DIR)

find_library(gmock_LIBRARY gmock)
mark_as_advanced(gmock_LIBRARY)

find_library(gmock_MAIN_LIBRARY gmock_main)
mark_as_advanced(gmock_MAIN_LIBRARY)

find_package_handle_standard_args(gtest DEFAULT_MSG
        gtest_INCLUDE_DIR
        gtest_LIBRARY
        gtest_MAIN_LIBRARY
        gmock_INCLUDE_DIR
        gmock_LIBRARY
        gmock_MAIN_LIBRARY
        )

set(URL https://github.com/google/googletest)
set(VERSION release-1.8.0)

set_package_properties(gtest
        PROPERTIES
        URL ${URL}
        DESCRIPTION "Unit test library"
        PURPOSE "commit: ${VERSION}"
        )
set_package_properties(gmock
        PROPERTIES
        URL ${URL}
        DESCRIPTION "Mock library"
        PURPOSE "commit: ${VERSION}"
        )

if (NOT gtest_FOUND)
    ExternalProject_Add(google_test
            GIT_REPOSITORY ${URL}
            GIT_TAG        ${VERSION}
            CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -Dgtest_force_shared_crt=ON
            -Dgtest_disable_pthreads=OFF
            BUILD_BYPRODUCTS ${EP_PREFIX}/src/google_test-build/googlemock/gtest/libgtest_main.a
            ${EP_PREFIX}/src/google_test-build/googlemock/gtest/libgtest.a
            ${EP_PREFIX}/src/google_test-build/googlemock/libgmock_main.a
            ${EP_PREFIX}/src/google_test-build/googlemock/libgmock.a
            INSTALL_COMMAND "" # remove install step
            UPDATE_COMMAND "" # remove update step
            TEST_COMMAND "" # remove test step
            )
    ExternalProject_Get_Property(google_test source_dir binary_dir)
    set(gtest_INCLUDE_DIR ${source_dir}/googletest/include)
    set(gmock_INCLUDE_DIR ${source_dir}/googlemock/include)

    set(gtest_MAIN_LIBRARY ${binary_dir}/googlemock/gtest/libgtest_main.a)
    set(gtest_LIBRARY ${binary_dir}/googlemock/gtest/libgtest.a)

    set(gmock_MAIN_LIBRARY ${binary_dir}/googlemock/libgmock_main.a)
    set(gmock_LIBRARY ${binary_dir}/googlemock/libgmock.a)

    file(MAKE_DIRECTORY ${gtest_INCLUDE_DIR})
    file(MAKE_DIRECTORY ${gmock_INCLUDE_DIR})

    add_dependencies(gtest google_test)
    add_dependencies(gmock google_test)
endif ()

set_target_properties(gtest PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${gtest_INCLUDE_DIR}
        INTERFACE_LINK_LIBRARIES "pthread;${gtest_MAIN_LIBRARY}"
        IMPORTED_LOCATION ${gtest_LIBRARY}
        )

set_target_properties(gmock PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${gmock_INCLUDE_DIR}
        INTERFACE_LINK_LIBRARIES "pthread;${gmock_MAIN_LIBRARY}"
        IMPORTED_LOCATION ${gmock_LIBRARY}
        )