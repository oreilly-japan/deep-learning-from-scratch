
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_subdirectory(
        primitive
)

add_executable(main
        main.cpp)
target_link_libraries(
        primitive
)

install ( TARGETS main
        RUNTIME DESTINATION bin
        CONFIGURATIONS Release
        COMPONENT binaries)