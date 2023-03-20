//===- EQueueOps.td ----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CDLT_DIALECT
#define CDLT_DIALECT
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// cdlt dialect definition.
//===----------------------------------------------------------------------===//

def Cdlt_Dialect : Dialect {
    let name = "cdlt";
    let summary = "codelet MLIR dialect.";
    let description = [{
        // This dialect is designed to provide abstractions of multi-kernel devices. 
        // It provides abstractions for device invocations on memories and computing kernels, 
        // allowing flexible device hierarchy and clear data flow.
        // The goal is to bridge the gap between low-level hardware model and high-level 
        // abstraction. 
        The dialect uses `cdlt` as its canonical prefix.
    }];
    let cppNamespace = "cdlt";
}





#endif // CDLT_DIALECT
