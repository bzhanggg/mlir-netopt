add_mlir_library(SingleProducerConstraint
    SingleProducerConstraint.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/src/analysis/SingleProducerConstraint

    LINK_LIBS PUBLIC
)