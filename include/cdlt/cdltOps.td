//===- EQueueOps.td ----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CDLT_OPS
#define CDLT_OPS
include "cdltDialect.td"
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
// Base cdlt operation definition.
//===----------------------------------------------------------------------===//

def StructureOpTrait : NativeOpTrait<"StructureOpTrait">;
def ControlOpTrait : NativeOpTrait<"ControlOpTrait">;
def AsyncOpTrait : NativeOpTrait<"AsyncOpTrait">;


// Base class for cdlt dialect ops.
class Cdlt_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cdlt_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// CodeletOp
//===----------------------------------------------------------------------===//


def CodeletOp : Cdlt_Op<"codelet", [AffineScope, AutomaticAllocationScope, CallableOpInterface, FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface, Symbol]>{
    let summary = "cdlt codelet operation";
    let description = [{
        Codelet is a small piece of code without branches or loops, which interacts with the system only via inputs and outputs. Once a codelet is launched on a divice, it will execute till the end without interruption. For the reasons above, the execution of a codelet is highly determined, making codelet the minimum schedulable unit.
        CodeletOp has a single region and a single block. 

        Example:
        cdlt.codelet @codelet_1()
        cdlt.codelet @codelet_2(%x: i64) -> (i64, i64){
            cdlt.return %x, %x: i64, i64
        }
    }];

    let arguments = (ins SymbolNameAttr:$sym_name,
                         TypeAttrOf<FunctionType>:$function_type,
                         OptionalAttr<StrAttr>:$sym_visibility);

    let regions = (region SizedRegion<1>:$region);

    let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>];
    let extraClassDeclaration = [{
    static CodeletOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {});
    static CodeletOp create(Location location, StringRef name, FunctionType type,
                         Operation::dialect_attr_range attrs);
    static CodeletOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<DictionaryAttr> argAttrs);

    /// Create a deep copy of this codelet and all of its blocks, remapping any
    /// operands that use values outside of the Codelet using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for Codelet arguments, these arguments are not
    /// included in the new Codelet. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    CodeletOp clone(BlockAndValueMapping &mapper);
    CodeletOp clone();

    /// Clone the internal blocks and attributes from this Codelet into dest.
    /// Any cloned blocks are appended to the back of dest. This Codelet
    /// asserts that the attributes of the current Codelet and dest are
    /// compatible.
    void cloneInto(CodeletOp dest, BlockAndValueMapping &mapper);

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// Codelet.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this Codelet.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this Codelet.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "cdlt"; }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Cdlt_Op<"return", [HasParent<"CodeletOp">,
                                MemRefsNormalizable, ReturnLike, Terminator]> {
  let summary = "Codelet return operation";
  let description = [{
    The `cdlt.return` operation represents a return operation within a codelet.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the codelet
    that contains the operation.

    Example:

    ```mlir
    cdlt.codelet @foo() : (i32, f8) {
      ...
      return %0, %1 : i32, f8
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, llvm::None);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExecOp
//===----------------------------------------------------------------------===//

def ExecOp : Cdlt_Op<"exec",
    [CallOpInterface, MemRefsNormalizable,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "codelet exec operation";
  let description = [{
    The `codelet.exec` operation represents a direct call to a codelet that is
    within the same symbol scope as the call. The operands and result types of
    the call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    Example:

    ```mlir
    %2 = codelet.exec @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "CodeletOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>];

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// SURgstOp
//===----------------------------------------------------------------------===//

def SURgst_Op : Cdlt_Op<"SURegst", [StructureOpTrait]>{
  let summary = "register a scheduling unit";
  let description = [{
    Register a scheduling unit. The arguments are:
    (1) the unique ID of the SU;
    (2) the freq of the SU, measured by codelets per second
    (3) the ID of the CUs controled by this SU. 

    Example:

    ```mlir
    cdlt.SURgst 1, 1,2,3,4
  }];

  let arguments = (ins I64Attr:$ID, F32Attr:$freq, I64SmallVectorArrayAttr:$CUs);
  let builders = [
    OpBuilder<(ins "int64_t":$ID,  "float32_t":$freq, "ArrayRef<int64_t>":$CUs)>
  ];
    
}

//===----------------------------------------------------------------------===//
// CURgstOp
//===----------------------------------------------------------------------===//

def CURgst_Op : Cdlt_Op<"CURegst", [StructureOpTrait]>{
  let summary = "register a computing unit";
  let description = [{
    Register a computing unit. The arguments are:
    (1) the unique ID of the CU;
    (2) the FLOPS of the CU

    Example:

    ```mlir
    cdlt.CURgst 1, 1.8E6
  }];

  let arguments = (ins I64Attr:$ID, F32Attr:$FLOPS);
  let builders = [
    OpBuilder<(ins "int64_t":$ID, "float32_t":$flops)>
  ];
    
}

//===----------------------------------------------------------------------===//
// MemRgstOp
//===----------------------------------------------------------------------===//

def MemRgst_Op : Cdlt_Op<"MemRgst", [StructureOpTrait]>{

  let summary = "Register a memory unit";

  let description = [{
    Register a memory unit. The arguments are:
    (1) the ID of the memory unit;
    (2) the start address of the memory space;
    (3) the end address of the memory space;
    (4) the category: local or global
    (5) if local, this argument indicates the CUs that have access to the memory;
        if global, the memory units that have access to the memory. 
  }];

  let arguments = (ins I64Attr:$ID, StrAttr:$start, StrAttr:$end, StrAttr:$scope, I64SmallVectorArrayAttr:$accessors);
  let results = (outs AnyMemRef);

  let builders = [
    OpBuilder<(ins "int64_t":$ID, "StringRef":$start, "StringRef":$end, "StringRef":$scope, "ArrayRef<int64_t>":$accessors)>
  ];

  
}

//===----------------------------------------------------------------------===//
// SpawnTPOp
//===----------------------------------------------------------------------===//
def SpawnTP_Op : Cdlt_Op<"spawn_TP", [AutomaticAllocationScope]>{
  let summary = "spawn a thread";

  let description = [{
    Thread is the minimum unit of resource allocation. A thread binds a SU and some CUs. Compution and memory access must take place inside a thread.
    The arguments are as follows:
    (1) the ID of the SU;
    (2) the IDs of the CUs;
  }];
  let arguments = (ins I64Attr:$SU, I64SmallVectorArrayAttr:$CUs);
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "int64_t":$SU, "ArrayRef<int64_t>":$CUs)>
  ];

}

//===----------------------------------------------------------------------===//
// KillTPOp
//===----------------------------------------------------------------------===//
def KillTP_Op : Cdlt_Op<"kill_TP", [AutomaticAllocationScope]>{
  let summary = "kill a thread";

  let description = [{
    Kill a thread and release the resource, including SU and CU.
    The arguments are as follows:
    (1) the ID of the SU;
    (2) the IDs of the CUs;
  }];
  let arguments = (ins I64Attr:$SU, I64SmallVectorArrayAttr:$CUs);
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "int64_t":$SU, "ArrayRef<int64_t>":$CUs)>
  ];

}


#endif // CDLT_OPS
