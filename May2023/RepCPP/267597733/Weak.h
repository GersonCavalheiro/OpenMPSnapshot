
#ifndef LLVM_CLANG_SEMA_WEAK_H
#define LLVM_CLANG_SEMA_WEAK_H

#include "clang/Basic/SourceLocation.h"

namespace clang {

class IdentifierInfo;

class WeakInfo {
IdentifierInfo *alias;  
SourceLocation loc;     
bool used;              
public:
WeakInfo()
: alias(nullptr), loc(SourceLocation()), used(false) {}
WeakInfo(IdentifierInfo *Alias, SourceLocation Loc)
: alias(Alias), loc(Loc), used(false) {}
inline IdentifierInfo * getAlias() const { return alias; }
inline SourceLocation getLocation() const { return loc; }
void setUsed(bool Used=true) { used = Used; }
inline bool getUsed() { return used; }
bool operator==(WeakInfo RHS) const {
return alias == RHS.getAlias() && loc == RHS.getLocation();
}
bool operator!=(WeakInfo RHS) const { return !(*this == RHS); }
};

} 

#endif 
