find_package(PackageHandleStandardArgs)

### Load External Project
include(ExternalProject)
include(FeatureSummary)

set_directory_properties(PROPERTIES
        EP_PREFIX ${EP_PREFIX}
        )

find_package(gtest)