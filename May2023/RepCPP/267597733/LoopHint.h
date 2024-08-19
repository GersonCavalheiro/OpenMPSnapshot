
#ifndef LLVM_CLANG_SEMA_LOOPHINT_H
#define LLVM_CLANG_SEMA_LOOPHINT_H

#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/ParsedAttr.h"

namespace clang {

struct LoopHint {
SourceRange Range;
IdentifierLoc *PragmaNameLoc;
IdentifierLoc *OptionLoc;
IdentifierLoc *StateLoc;
Expr *ValueExpr;

LoopHint()
: PragmaNameLoc(nullptr), OptionLoc(nullptr), StateLoc(nullptr),
ValueExpr(nullptr) {}
};

} 

#endif 
