add_executable(
        ndarray_test ndarray_test.cpp)
target_link_libraries(ndarray_test
        gtest)

add_test(
        NAME ndarray_test
        COMMAND $<TARGET_FILE:ndarray_test>)
