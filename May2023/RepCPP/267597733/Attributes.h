
#ifndef LLVM_CLANG_BASIC_ATTRIBUTES_H
#define LLVM_CLANG_BASIC_ATTRIBUTES_H

#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"

namespace clang {

class IdentifierInfo;

enum class AttrSyntax {
GNU,
Declspec,
Microsoft,
CXX,
C,
Pragma
};

int hasAttribute(AttrSyntax Syntax, const IdentifierInfo *Scope,
const IdentifierInfo *Attr, const TargetInfo &Target,
const LangOptions &LangOpts);

} 

#endif 
