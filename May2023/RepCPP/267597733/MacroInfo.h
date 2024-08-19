
#ifndef LLVM_CLANG_LEX_MACROINFO_H
#define LLVM_CLANG_LEX_MACROINFO_H

#include "clang/Lex/Token.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include <algorithm>
#include <cassert>

namespace clang {

class DefMacroDirective;
class IdentifierInfo;
class Module;
class Preprocessor;
class SourceManager;

class MacroInfo {

SourceLocation Location;

SourceLocation EndLocation;

IdentifierInfo **ParameterList = nullptr;

unsigned NumParameters = 0;

SmallVector<Token, 8> ReplacementTokens;

mutable unsigned DefinitionLength;
mutable bool IsDefinitionLengthCached : 1;

bool IsFunctionLike : 1;

bool IsC99Varargs : 1;

bool IsGNUVarargs : 1;

bool IsBuiltinMacro : 1;

bool HasCommaPasting : 1;


bool IsDisabled : 1;

bool IsUsed : 1;

bool IsAllowRedefinitionsWithoutWarning : 1;

bool IsWarnIfUnused : 1;

bool UsedForHeaderGuard : 1;

MacroInfo(SourceLocation DefLoc);
~MacroInfo() = default;

public:
SourceLocation getDefinitionLoc() const { return Location; }

void setDefinitionEndLoc(SourceLocation EndLoc) { EndLocation = EndLoc; }

SourceLocation getDefinitionEndLoc() const { return EndLocation; }

unsigned getDefinitionLength(const SourceManager &SM) const {
if (IsDefinitionLengthCached)
return DefinitionLength;
return getDefinitionLengthSlow(SM);
}

bool isIdenticalTo(const MacroInfo &Other, Preprocessor &PP,
bool Syntactically) const;

void setIsBuiltinMacro(bool Val = true) { IsBuiltinMacro = Val; }

void setIsUsed(bool Val) { IsUsed = Val; }

void setIsAllowRedefinitionsWithoutWarning(bool Val) {
IsAllowRedefinitionsWithoutWarning = Val;
}

void setIsWarnIfUnused(bool val) { IsWarnIfUnused = val; }

void setParameterList(ArrayRef<IdentifierInfo *> List,
llvm::BumpPtrAllocator &PPAllocator) {
assert(ParameterList == nullptr && NumParameters == 0 &&
"Parameter list already set!");
if (List.empty())
return;

NumParameters = List.size();
ParameterList = PPAllocator.Allocate<IdentifierInfo *>(List.size());
std::copy(List.begin(), List.end(), ParameterList);
}

using param_iterator = IdentifierInfo *const *;
bool param_empty() const { return NumParameters == 0; }
param_iterator param_begin() const { return ParameterList; }
param_iterator param_end() const { return ParameterList + NumParameters; }
unsigned getNumParams() const { return NumParameters; }
ArrayRef<const IdentifierInfo *> params() const {
return ArrayRef<const IdentifierInfo *>(ParameterList, NumParameters);
}

int getParameterNum(const IdentifierInfo *Arg) const {
for (param_iterator I = param_begin(), E = param_end(); I != E; ++I)
if (*I == Arg)
return I - param_begin();
return -1;
}

void setIsFunctionLike() { IsFunctionLike = true; }
bool isFunctionLike() const { return IsFunctionLike; }
bool isObjectLike() const { return !IsFunctionLike; }

void setIsC99Varargs() { IsC99Varargs = true; }
void setIsGNUVarargs() { IsGNUVarargs = true; }
bool isC99Varargs() const { return IsC99Varargs; }
bool isGNUVarargs() const { return IsGNUVarargs; }
bool isVariadic() const { return IsC99Varargs | IsGNUVarargs; }

bool isBuiltinMacro() const { return IsBuiltinMacro; }

bool hasCommaPasting() const { return HasCommaPasting; }
void setHasCommaPasting() { HasCommaPasting = true; }

bool isUsed() const { return IsUsed; }

bool isAllowRedefinitionsWithoutWarning() const {
return IsAllowRedefinitionsWithoutWarning;
}

bool isWarnIfUnused() const { return IsWarnIfUnused; }

unsigned getNumTokens() const { return ReplacementTokens.size(); }

const Token &getReplacementToken(unsigned Tok) const {
assert(Tok < ReplacementTokens.size() && "Invalid token #");
return ReplacementTokens[Tok];
}

using tokens_iterator = SmallVectorImpl<Token>::const_iterator;

tokens_iterator tokens_begin() const { return ReplacementTokens.begin(); }
tokens_iterator tokens_end() const { return ReplacementTokens.end(); }
bool tokens_empty() const { return ReplacementTokens.empty(); }
ArrayRef<Token> tokens() const { return ReplacementTokens; }

void AddTokenToBody(const Token &Tok) {
assert(
!IsDefinitionLengthCached &&
"Changing replacement tokens after definition length got calculated");
ReplacementTokens.push_back(Tok);
}

bool isEnabled() const { return !IsDisabled; }

void EnableMacro() {
assert(IsDisabled && "Cannot enable an already-enabled macro!");
IsDisabled = false;
}

void DisableMacro() {
assert(!IsDisabled && "Cannot disable an already-disabled macro!");
IsDisabled = true;
}

bool isUsedForHeaderGuard() const { return UsedForHeaderGuard; }

void setUsedForHeaderGuard(bool Val) { UsedForHeaderGuard = Val; }

void dump() const;

private:
friend class Preprocessor;

unsigned getDefinitionLengthSlow(const SourceManager &SM) const;
};

class MacroDirective {
public:
enum Kind {
MD_Define,
MD_Undefine,
MD_Visibility
};

protected:
MacroDirective *Previous = nullptr;

SourceLocation Loc;

unsigned MDKind : 2;

unsigned IsFromPCH : 1;


unsigned IsPublic : 1;

MacroDirective(Kind K, SourceLocation Loc)
: Loc(Loc), MDKind(K), IsFromPCH(false), IsPublic(true) {}

public:
Kind getKind() const { return Kind(MDKind); }

SourceLocation getLocation() const { return Loc; }

void setPrevious(MacroDirective *Prev) { Previous = Prev; }

const MacroDirective *getPrevious() const { return Previous; }

MacroDirective *getPrevious() { return Previous; }

bool isFromPCH() const { return IsFromPCH; }

void setIsFromPCH() { IsFromPCH = true; }

class DefInfo {
DefMacroDirective *DefDirective = nullptr;
SourceLocation UndefLoc;
bool IsPublic = true;

public:
DefInfo() = default;
DefInfo(DefMacroDirective *DefDirective, SourceLocation UndefLoc,
bool isPublic)
: DefDirective(DefDirective), UndefLoc(UndefLoc), IsPublic(isPublic) {}

const DefMacroDirective *getDirective() const { return DefDirective; }
DefMacroDirective *getDirective() { return DefDirective; }

inline SourceLocation getLocation() const;
inline MacroInfo *getMacroInfo();

const MacroInfo *getMacroInfo() const {
return const_cast<DefInfo *>(this)->getMacroInfo();
}

SourceLocation getUndefLocation() const { return UndefLoc; }
bool isUndefined() const { return UndefLoc.isValid(); }

bool isPublic() const { return IsPublic; }

bool isValid() const { return DefDirective != nullptr; }
bool isInvalid() const { return !isValid(); }

explicit operator bool() const { return isValid(); }

inline DefInfo getPreviousDefinition();

const DefInfo getPreviousDefinition() const {
return const_cast<DefInfo *>(this)->getPreviousDefinition();
}
};

DefInfo getDefinition();
const DefInfo getDefinition() const {
return const_cast<MacroDirective *>(this)->getDefinition();
}

bool isDefined() const {
if (const DefInfo Def = getDefinition())
return !Def.isUndefined();
return false;
}

const MacroInfo *getMacroInfo() const {
return getDefinition().getMacroInfo();
}
MacroInfo *getMacroInfo() { return getDefinition().getMacroInfo(); }

const DefInfo findDirectiveAtLoc(SourceLocation L, SourceManager &SM) const;

void dump() const;

static bool classof(const MacroDirective *) { return true; }
};

class DefMacroDirective : public MacroDirective {
MacroInfo *Info;

public:
DefMacroDirective(MacroInfo *MI, SourceLocation Loc)
: MacroDirective(MD_Define, Loc), Info(MI) {
assert(MI && "MacroInfo is null");
}
explicit DefMacroDirective(MacroInfo *MI)
: DefMacroDirective(MI, MI->getDefinitionLoc()) {}

const MacroInfo *getInfo() const { return Info; }
MacroInfo *getInfo() { return Info; }

static bool classof(const MacroDirective *MD) {
return MD->getKind() == MD_Define;
}

static bool classof(const DefMacroDirective *) { return true; }
};

class UndefMacroDirective : public MacroDirective {
public:
explicit UndefMacroDirective(SourceLocation UndefLoc)
: MacroDirective(MD_Undefine, UndefLoc) {
assert(UndefLoc.isValid() && "Invalid UndefLoc!");
}

static bool classof(const MacroDirective *MD) {
return MD->getKind() == MD_Undefine;
}

static bool classof(const UndefMacroDirective *) { return true; }
};

class VisibilityMacroDirective : public MacroDirective {
public:
explicit VisibilityMacroDirective(SourceLocation Loc, bool Public)
: MacroDirective(MD_Visibility, Loc) {
IsPublic = Public;
}

bool isPublic() const { return IsPublic; }

static bool classof(const MacroDirective *MD) {
return MD->getKind() == MD_Visibility;
}

static bool classof(const VisibilityMacroDirective *) { return true; }
};

inline SourceLocation MacroDirective::DefInfo::getLocation() const {
if (isInvalid())
return {};
return DefDirective->getLocation();
}

inline MacroInfo *MacroDirective::DefInfo::getMacroInfo() {
if (isInvalid())
return nullptr;
return DefDirective->getInfo();
}

inline MacroDirective::DefInfo
MacroDirective::DefInfo::getPreviousDefinition() {
if (isInvalid() || DefDirective->getPrevious() == nullptr)
return {};
return DefDirective->getPrevious()->getDefinition();
}

class ModuleMacro : public llvm::FoldingSetNode {
friend class Preprocessor;

IdentifierInfo *II;

MacroInfo *Macro;

Module *OwningModule;

unsigned NumOverriddenBy = 0;

unsigned NumOverrides;

ModuleMacro(Module *OwningModule, IdentifierInfo *II, MacroInfo *Macro,
ArrayRef<ModuleMacro *> Overrides)
: II(II), Macro(Macro), OwningModule(OwningModule),
NumOverrides(Overrides.size()) {
std::copy(Overrides.begin(), Overrides.end(),
reinterpret_cast<ModuleMacro **>(this + 1));
}

public:
static ModuleMacro *create(Preprocessor &PP, Module *OwningModule,
IdentifierInfo *II, MacroInfo *Macro,
ArrayRef<ModuleMacro *> Overrides);

void Profile(llvm::FoldingSetNodeID &ID) const {
return Profile(ID, OwningModule, II);
}

static void Profile(llvm::FoldingSetNodeID &ID, Module *OwningModule,
IdentifierInfo *II) {
ID.AddPointer(OwningModule);
ID.AddPointer(II);
}

IdentifierInfo *getName() const { return II; }

Module *getOwningModule() const { return OwningModule; }

MacroInfo *getMacroInfo() const { return Macro; }

using overrides_iterator = ModuleMacro *const *;

overrides_iterator overrides_begin() const {
return reinterpret_cast<overrides_iterator>(this + 1);
}

overrides_iterator overrides_end() const {
return overrides_begin() + NumOverrides;
}

ArrayRef<ModuleMacro *> overrides() const {
return llvm::makeArrayRef(overrides_begin(), overrides_end());
}

unsigned getNumOverridingMacros() const { return NumOverriddenBy; }
};

class MacroDefinition {
llvm::PointerIntPair<DefMacroDirective *, 1, bool> LatestLocalAndAmbiguous;
ArrayRef<ModuleMacro *> ModuleMacros;

public:
MacroDefinition() = default;
MacroDefinition(DefMacroDirective *MD, ArrayRef<ModuleMacro *> MMs,
bool IsAmbiguous)
: LatestLocalAndAmbiguous(MD, IsAmbiguous), ModuleMacros(MMs) {}

explicit operator bool() const {
return getLocalDirective() || !ModuleMacros.empty();
}

MacroInfo *getMacroInfo() const {
if (!ModuleMacros.empty())
return ModuleMacros.back()->getMacroInfo();
if (auto *MD = getLocalDirective())
return MD->getMacroInfo();
return nullptr;
}

bool isAmbiguous() const { return LatestLocalAndAmbiguous.getInt(); }

DefMacroDirective *getLocalDirective() const {
return LatestLocalAndAmbiguous.getPointer();
}

ArrayRef<ModuleMacro *> getModuleMacros() const { return ModuleMacros; }

template <typename Fn> void forAllDefinitions(Fn F) const {
if (auto *MD = getLocalDirective())
F(MD->getMacroInfo());
for (auto *MM : getModuleMacros())
F(MM->getMacroInfo());
}
};

} 

#endif 
