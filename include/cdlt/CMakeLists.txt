# add_custom_target(cdltOps)
# set_target_properties(cdltOps PROPERTIES FOLDER cdlt)
# include_directories(${PROJECT_SOURCE_DIR})
# add_dependencies(Cdlt cdlt)

set(LLVM_TARGET_DEFINITIONS cdltDialect.td)
mlir_tablegen(cdltDialect.h.inc -gen-dialect-decls)
mlir_tablegen(cdltDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS cdltOps.td)
mlir_tablegen(cdltOps.h.inc -gen-op-decls)
mlir_tablegen(cdltOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(cdltbase)

add_mlir_dialect(cdltOPs cdlt)
add_mlir_doc(cdltDialect cdltDialect cdlt/ -gen-dialect-doc)
add_mlir_doc(cdltOps cdltOps cdlt/ -gen-op-doc )

set(LLVM_TARGET_DEFINITIONS CdltPasses.td)
mlir_tablegen(CdltPasses.h.inc -gen-pass-decls -name Conversion)
mlir_tablegen(CdltPasses.capi.h.inc -gen-pass-capi-header --prefix Conversion)
mlir_tablegen(CdltPasses.capi.cpp.inc -gen-pass-capi-impl --prefix Conversion)
add_public_tablegen_target(MLIRConversionPassIncGen)

add_mlir_doc(Passes ConversionPasses ./ -gen-pass-doc)








