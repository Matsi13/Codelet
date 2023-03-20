#ifndef CDLT_CONVERSION_PASSES
#define CDLT_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// AffineToStandard
//===----------------------------------------------------------------------===//

def ConvertCdltToFunc : Pass<"cdlt-func"> {
  let summary = "lower cdlt ops to func dialect";
  let description = [{

    Convert operations from the cdlt dialect into operations from func dialect.

  }];
  let constructor = "mlir::createCdltToFuncPass()";
  let dependentDialects = [
    "func::FuncDialect"
  ];
}


def ConvertCdltToGPU : Pass<"cdlt-gpu"> {
  let summary = "lower cdlt ops to gpu dialect";
  let description = [{

    Convert operations from the cdlt dialect into operations from gpu dialect.

  }];
  let constructor = "mlir::createCdltToGPUPass()";
  let dependentDialects = [
    "gpu::GPUDialect"
  ];
}



#endif //CDLT_CONVERSION_PASSES
