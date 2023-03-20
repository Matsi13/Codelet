#ifndef CDLTDIALECT_H
#define CDLTDIALECT_H

#include "mlir/IR/Dialect.h"

using namespace mlir;
namespace cdlt {

// The Dialect
class CdltDialect : public mlir::Dialect {
public:
  explicit CdltDialect(mlir::MLIRContext *ctx);
  static StringRef getDialectNamespace() { return "cdlt";}
  void initialize();

};


}


#endif // CDLTDIALECT_H