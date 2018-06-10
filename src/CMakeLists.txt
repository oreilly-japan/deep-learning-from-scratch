
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_executable(main
        main.cpp)

install ( TARGETS main
        RUNTIME DESTINATION bin
        CONFIGURATIONS Release
        COMPONENT binaries)