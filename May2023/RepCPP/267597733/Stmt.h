
#ifndef LLVM_CLANG_AST_STMT_H
#define LLVM_CLANG_AST_STMT_H

#include "clang/AST/DeclGroup.h"
#include "clang/AST/StmtIterator.h"
#include "clang/Basic/CapturedStmt.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <string>

namespace llvm {

class FoldingSetNodeID;

} 

namespace clang {

class ASTContext;
class Attr;
class CapturedDecl;
class Decl;
class Expr;
class LabelDecl;
class ODRHash;
class PrinterHelper;
struct PrintingPolicy;
class RecordDecl;
class SourceManager;
class StringLiteral;
class Token;
class VarDecl;


class alignas(void *) Stmt {
public:
enum StmtClass {
NoStmtClass = 0,
#define STMT(CLASS, PARENT) CLASS##Class,
#define STMT_RANGE(BASE, FIRST, LAST) \
first##BASE##Constant=FIRST##Class, last##BASE##Constant=LAST##Class,
#define LAST_STMT_RANGE(BASE, FIRST, LAST) \
first##BASE##Constant=FIRST##Class, last##BASE##Constant=LAST##Class
#define ABSTRACT_STMT(STMT)
#include "clang/AST/StmtNodes.inc"
};

protected:
friend class ASTStmtReader;
friend class ASTStmtWriter;

void *operator new(size_t bytes) noexcept {
llvm_unreachable("Stmts cannot be allocated with regular 'new'.");
}

void operator delete(void *data) noexcept {
llvm_unreachable("Stmts cannot be released with regular 'delete'.");
}

class StmtBitfields {
friend class Stmt;

unsigned sClass : 8;
};
enum { NumStmtBits = 8 };

class CompoundStmtBitfields {
friend class CompoundStmt;

unsigned : NumStmtBits;

unsigned NumStmts : 32 - NumStmtBits;
};

class IfStmtBitfields {
friend class IfStmt;

unsigned : NumStmtBits;

unsigned IsConstexpr : 1;
};

class ExprBitfields {
friend class ASTStmtReader; 
friend class AtomicExpr; 
friend class BlockDeclRefExpr; 
friend class CallExpr; 
friend class CXXConstructExpr; 
friend class CXXDependentScopeMemberExpr; 
friend class CXXNewExpr; 
friend class CXXUnresolvedConstructExpr; 
friend class DeclRefExpr; 
friend class DependentScopeDeclRefExpr; 
friend class DesignatedInitExpr; 
friend class Expr;
friend class InitListExpr; 
friend class ObjCArrayLiteral; 
friend class ObjCDictionaryLiteral; 
friend class ObjCMessageExpr; 
friend class OffsetOfExpr; 
friend class OpaqueValueExpr; 
friend class OverloadExpr; 
friend class ParenListExpr; 
friend class PseudoObjectExpr; 
friend class ShuffleVectorExpr; 

unsigned : NumStmtBits;

unsigned ValueKind : 2;
unsigned ObjectKind : 3;
unsigned TypeDependent : 1;
unsigned ValueDependent : 1;
unsigned InstantiationDependent : 1;
unsigned ContainsUnexpandedParameterPack : 1;
};
enum { NumExprBits = 17 };

class CharacterLiteralBitfields {
friend class CharacterLiteral;

unsigned : NumExprBits;

unsigned Kind : 3;
};

enum APFloatSemantics {
IEEEhalf,
IEEEsingle,
IEEEdouble,
x87DoubleExtended,
IEEEquad,
PPCDoubleDouble
};

class FloatingLiteralBitfields {
friend class FloatingLiteral;

unsigned : NumExprBits;

unsigned Semantics : 3; 
unsigned IsExact : 1;
};

class UnaryExprOrTypeTraitExprBitfields {
friend class UnaryExprOrTypeTraitExpr;

unsigned : NumExprBits;

unsigned Kind : 2;
unsigned IsType : 1; 
};

class DeclRefExprBitfields {
friend class ASTStmtReader; 
friend class DeclRefExpr;

unsigned : NumExprBits;

unsigned HasQualifier : 1;
unsigned HasTemplateKWAndArgsInfo : 1;
unsigned HasFoundDecl : 1;
unsigned HadMultipleCandidates : 1;
unsigned RefersToEnclosingVariableOrCapture : 1;
};

class CastExprBitfields {
friend class CastExpr;
friend class ImplicitCastExpr;

unsigned : NumExprBits;

unsigned Kind : 6;
unsigned PartOfExplicitCast : 1; 
unsigned BasePathIsEmpty : 1;
};

class CallExprBitfields {
friend class CallExpr;

unsigned : NumExprBits;

unsigned NumPreArgs : 1;
};

class ExprWithCleanupsBitfields {
friend class ASTStmtReader; 
friend class ExprWithCleanups;

unsigned : NumExprBits;

unsigned CleanupsHaveSideEffects : 1;

unsigned NumObjects : 32 - 1 - NumExprBits;
};

class PseudoObjectExprBitfields {
friend class ASTStmtReader; 
friend class PseudoObjectExpr;

unsigned : NumExprBits;

unsigned NumSubExprs : 8;
unsigned ResultIndex : 32 - 8 - NumExprBits;
};

class OpaqueValueExprBitfields {
friend class OpaqueValueExpr;

unsigned : NumExprBits;

unsigned IsUnique : 1;
};

class ObjCIndirectCopyRestoreExprBitfields {
friend class ObjCIndirectCopyRestoreExpr;

unsigned : NumExprBits;

unsigned ShouldCopy : 1;
};

class InitListExprBitfields {
friend class InitListExpr;

unsigned : NumExprBits;

unsigned HadArrayRangeDesignator : 1;
};

class TypeTraitExprBitfields {
friend class ASTStmtReader;
friend class ASTStmtWriter;
friend class TypeTraitExpr;

unsigned : NumExprBits;

unsigned Kind : 8;

unsigned Value : 1;

unsigned NumArgs : 32 - 8 - 1 - NumExprBits;
};

class CoawaitExprBitfields {
friend class CoawaitExpr;

unsigned : NumExprBits;

unsigned IsImplicit : 1;
};

union {
StmtBitfields StmtBits;
CompoundStmtBitfields CompoundStmtBits;
IfStmtBitfields IfStmtBits;
ExprBitfields ExprBits;
CharacterLiteralBitfields CharacterLiteralBits;
FloatingLiteralBitfields FloatingLiteralBits;
UnaryExprOrTypeTraitExprBitfields UnaryExprOrTypeTraitExprBits;
DeclRefExprBitfields DeclRefExprBits;
CastExprBitfields CastExprBits;
CallExprBitfields CallExprBits;
ExprWithCleanupsBitfields ExprWithCleanupsBits;
PseudoObjectExprBitfields PseudoObjectExprBits;
OpaqueValueExprBitfields OpaqueValueExprBits;
ObjCIndirectCopyRestoreExprBitfields ObjCIndirectCopyRestoreExprBits;
InitListExprBitfields InitListExprBits;
TypeTraitExprBitfields TypeTraitExprBits;
CoawaitExprBitfields CoawaitBits;
};

public:
void* operator new(size_t bytes, const ASTContext& C,
unsigned alignment = 8);

void* operator new(size_t bytes, const ASTContext* C,
unsigned alignment = 8) {
return operator new(bytes, *C, alignment);
}

void *operator new(size_t bytes, void *mem) noexcept { return mem; }

void operator delete(void *, const ASTContext &, unsigned) noexcept {}
void operator delete(void *, const ASTContext *, unsigned) noexcept {}
void operator delete(void *, size_t) noexcept {}
void operator delete(void *, void *) noexcept {}

public:
struct EmptyShell {};

protected:
struct ExprIterator
: llvm::iterator_adaptor_base<ExprIterator, Stmt **,
std::random_access_iterator_tag, Expr *> {
ExprIterator() : iterator_adaptor_base(nullptr) {}
ExprIterator(Stmt **I) : iterator_adaptor_base(I) {}

reference operator*() const {
assert((*I)->getStmtClass() >= firstExprConstant &&
(*I)->getStmtClass() <= lastExprConstant);
return *reinterpret_cast<Expr **>(I);
}
};

struct ConstExprIterator
: llvm::iterator_adaptor_base<ConstExprIterator, const Stmt *const *,
std::random_access_iterator_tag,
const Expr *const> {
ConstExprIterator() : iterator_adaptor_base(nullptr) {}
ConstExprIterator(const Stmt *const *I) : iterator_adaptor_base(I) {}

reference operator*() const {
assert((*I)->getStmtClass() >= firstExprConstant &&
(*I)->getStmtClass() <= lastExprConstant);
return *reinterpret_cast<const Expr *const *>(I);
}
};

private:
static bool StatisticsEnabled;

protected:
explicit Stmt(StmtClass SC, EmptyShell) : Stmt(SC) {}

public:
Stmt(StmtClass SC) {
static_assert(sizeof(*this) == sizeof(void *),
"changing bitfields changed sizeof(Stmt)");
static_assert(sizeof(*this) % alignof(void *) == 0,
"Insufficient alignment!");
StmtBits.sClass = SC;
if (StatisticsEnabled) Stmt::addStmtClass(SC);
}

StmtClass getStmtClass() const {
return static_cast<StmtClass>(StmtBits.sClass);
}

const char *getStmtClassName() const;

SourceRange getSourceRange() const LLVM_READONLY;
SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY;
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY;

static void addStmtClass(const StmtClass s);
static void EnableStatistics();
static void PrintStats();

void dump() const;
void dump(SourceManager &SM) const;
void dump(raw_ostream &OS, SourceManager &SM) const;
void dump(raw_ostream &OS) const;

void dumpColor() const;

void dumpPretty(const ASTContext &Context) const;
void printPretty(raw_ostream &OS, PrinterHelper *Helper,
const PrintingPolicy &Policy, unsigned Indentation = 0,
const ASTContext *Context = nullptr) const;

void viewAST() const;

Stmt *IgnoreImplicit();
const Stmt *IgnoreImplicit() const {
return const_cast<Stmt *>(this)->IgnoreImplicit();
}

Stmt *IgnoreContainers(bool IgnoreCaptured = false);
const Stmt *IgnoreContainers(bool IgnoreCaptured = false) const {
return const_cast<Stmt *>(this)->IgnoreContainers(IgnoreCaptured);
}

const Stmt *stripLabelLikeStatements() const;
Stmt *stripLabelLikeStatements() {
return const_cast<Stmt*>(
const_cast<const Stmt*>(this)->stripLabelLikeStatements());
}

using child_iterator = StmtIterator;
using const_child_iterator = ConstStmtIterator;

using child_range = llvm::iterator_range<child_iterator>;
using const_child_range = llvm::iterator_range<const_child_iterator>;

child_range children();

const_child_range children() const {
auto Children = const_cast<Stmt *>(this)->children();
return const_child_range(Children.begin(), Children.end());
}

child_iterator child_begin() { return children().begin(); }
child_iterator child_end() { return children().end(); }

const_child_iterator child_begin() const { return children().begin(); }
const_child_iterator child_end() const { return children().end(); }

void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Context,
bool Canonical) const;

void ProcessODRHash(llvm::FoldingSetNodeID &ID, ODRHash& Hash) const;
};

class DeclStmt : public Stmt {
DeclGroupRef DG;
SourceLocation StartLoc, EndLoc;

public:
DeclStmt(DeclGroupRef dg, SourceLocation startLoc, SourceLocation endLoc)
: Stmt(DeclStmtClass), DG(dg), StartLoc(startLoc), EndLoc(endLoc) {}

explicit DeclStmt(EmptyShell Empty) : Stmt(DeclStmtClass, Empty) {}

bool isSingleDecl() const {
return DG.isSingleDecl();
}

const Decl *getSingleDecl() const { return DG.getSingleDecl(); }
Decl *getSingleDecl() { return DG.getSingleDecl(); }

const DeclGroupRef getDeclGroup() const { return DG; }
DeclGroupRef getDeclGroup() { return DG; }
void setDeclGroup(DeclGroupRef DGR) { DG = DGR; }

SourceLocation getStartLoc() const LLVM_READONLY { return getBeginLoc(); }
void setStartLoc(SourceLocation L) { StartLoc = L; }
SourceLocation getEndLoc() const { return EndLoc; }
void setEndLoc(SourceLocation L) { EndLoc = L; }

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return StartLoc; }
SourceLocation getLocEnd() const LLVM_READONLY { return EndLoc; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == DeclStmtClass;
}

child_range children() {
return child_range(child_iterator(DG.begin(), DG.end()),
child_iterator(DG.end(), DG.end()));
}

using decl_iterator = DeclGroupRef::iterator;
using const_decl_iterator = DeclGroupRef::const_iterator;
using decl_range = llvm::iterator_range<decl_iterator>;
using decl_const_range = llvm::iterator_range<const_decl_iterator>;

decl_range decls() { return decl_range(decl_begin(), decl_end()); }

decl_const_range decls() const {
return decl_const_range(decl_begin(), decl_end());
}

decl_iterator decl_begin() { return DG.begin(); }
decl_iterator decl_end() { return DG.end(); }
const_decl_iterator decl_begin() const { return DG.begin(); }
const_decl_iterator decl_end() const { return DG.end(); }

using reverse_decl_iterator = std::reverse_iterator<decl_iterator>;

reverse_decl_iterator decl_rbegin() {
return reverse_decl_iterator(decl_end());
}

reverse_decl_iterator decl_rend() {
return reverse_decl_iterator(decl_begin());
}
};

class NullStmt : public Stmt {
SourceLocation SemiLoc;

bool HasLeadingEmptyMacro = false;

public:
friend class ASTStmtReader;
friend class ASTStmtWriter;

NullStmt(SourceLocation L, bool hasLeadingEmptyMacro = false)
: Stmt(NullStmtClass), SemiLoc(L),
HasLeadingEmptyMacro(hasLeadingEmptyMacro) {}

explicit NullStmt(EmptyShell Empty) : Stmt(NullStmtClass, Empty) {}

SourceLocation getSemiLoc() const { return SemiLoc; }
void setSemiLoc(SourceLocation L) { SemiLoc = L; }

bool hasLeadingEmptyMacro() const { return HasLeadingEmptyMacro; }

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return SemiLoc; }
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY { return SemiLoc; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == NullStmtClass;
}

child_range children() {
return child_range(child_iterator(), child_iterator());
}
};

class CompoundStmt final : public Stmt,
private llvm::TrailingObjects<CompoundStmt, Stmt *> {
friend class ASTStmtReader;
friend TrailingObjects;

SourceLocation LBraceLoc, RBraceLoc;

CompoundStmt(ArrayRef<Stmt *> Stmts, SourceLocation LB, SourceLocation RB);
explicit CompoundStmt(EmptyShell Empty) : Stmt(CompoundStmtClass, Empty) {}

void setStmts(ArrayRef<Stmt *> Stmts);

public:
static CompoundStmt *Create(const ASTContext &C, ArrayRef<Stmt *> Stmts,
SourceLocation LB, SourceLocation RB);

explicit CompoundStmt(SourceLocation Loc)
: Stmt(CompoundStmtClass), LBraceLoc(Loc), RBraceLoc(Loc) {
CompoundStmtBits.NumStmts = 0;
}

static CompoundStmt *CreateEmpty(const ASTContext &C, unsigned NumStmts);

bool body_empty() const { return CompoundStmtBits.NumStmts == 0; }
unsigned size() const { return CompoundStmtBits.NumStmts; }

using body_iterator = Stmt **;
using body_range = llvm::iterator_range<body_iterator>;

body_range body() { return body_range(body_begin(), body_end()); }
body_iterator body_begin() { return getTrailingObjects<Stmt *>(); }
body_iterator body_end() { return body_begin() + size(); }
Stmt *body_front() { return !body_empty() ? body_begin()[0] : nullptr; }

Stmt *body_back() {
return !body_empty() ? body_begin()[size() - 1] : nullptr;
}

void setLastStmt(Stmt *S) {
assert(!body_empty() && "setLastStmt");
body_begin()[size() - 1] = S;
}

using const_body_iterator = Stmt* const *;
using body_const_range = llvm::iterator_range<const_body_iterator>;

body_const_range body() const {
return body_const_range(body_begin(), body_end());
}

const_body_iterator body_begin() const {
return getTrailingObjects<Stmt *>();
}

const_body_iterator body_end() const { return body_begin() + size(); }

const Stmt *body_front() const {
return !body_empty() ? body_begin()[0] : nullptr;
}

const Stmt *body_back() const {
return !body_empty() ? body_begin()[size() - 1] : nullptr;
}

using reverse_body_iterator = std::reverse_iterator<body_iterator>;

reverse_body_iterator body_rbegin() {
return reverse_body_iterator(body_end());
}

reverse_body_iterator body_rend() {
return reverse_body_iterator(body_begin());
}

using const_reverse_body_iterator =
std::reverse_iterator<const_body_iterator>;

const_reverse_body_iterator body_rbegin() const {
return const_reverse_body_iterator(body_end());
}

const_reverse_body_iterator body_rend() const {
return const_reverse_body_iterator(body_begin());
}

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return LBraceLoc; }
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY { return RBraceLoc; }

SourceLocation getLBracLoc() const { return LBraceLoc; }
SourceLocation getRBracLoc() const { return RBraceLoc; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == CompoundStmtClass;
}

child_range children() { return child_range(body_begin(), body_end()); }

const_child_range children() const {
return const_child_range(body_begin(), body_end());
}
};

class SwitchCase : public Stmt {
protected:
SwitchCase *NextSwitchCase = nullptr;
SourceLocation KeywordLoc;
SourceLocation ColonLoc;

SwitchCase(StmtClass SC, SourceLocation KWLoc, SourceLocation ColonLoc)
: Stmt(SC), KeywordLoc(KWLoc), ColonLoc(ColonLoc) {}

SwitchCase(StmtClass SC, EmptyShell) : Stmt(SC) {}

public:
const SwitchCase *getNextSwitchCase() const { return NextSwitchCase; }

SwitchCase *getNextSwitchCase() { return NextSwitchCase; }

void setNextSwitchCase(SwitchCase *SC) { NextSwitchCase = SC; }

SourceLocation getKeywordLoc() const { return KeywordLoc; }
void setKeywordLoc(SourceLocation L) { KeywordLoc = L; }
SourceLocation getColonLoc() const { return ColonLoc; }
void setColonLoc(SourceLocation L) { ColonLoc = L; }

Stmt *getSubStmt();
const Stmt *getSubStmt() const {
return const_cast<SwitchCase*>(this)->getSubStmt();
}

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return KeywordLoc; }
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY;

static bool classof(const Stmt *T) {
return T->getStmtClass() == CaseStmtClass ||
T->getStmtClass() == DefaultStmtClass;
}
};

class CaseStmt : public SwitchCase {
SourceLocation EllipsisLoc;
enum { LHS, RHS, SUBSTMT, END_EXPR };
Stmt* SubExprs[END_EXPR];  

public:
CaseStmt(Expr *lhs, Expr *rhs, SourceLocation caseLoc,
SourceLocation ellipsisLoc, SourceLocation colonLoc)
: SwitchCase(CaseStmtClass, caseLoc, colonLoc) {
SubExprs[SUBSTMT] = nullptr;
SubExprs[LHS] = reinterpret_cast<Stmt*>(lhs);
SubExprs[RHS] = reinterpret_cast<Stmt*>(rhs);
EllipsisLoc = ellipsisLoc;
}

explicit CaseStmt(EmptyShell Empty) : SwitchCase(CaseStmtClass, Empty) {}

SourceLocation getCaseLoc() const { return KeywordLoc; }
void setCaseLoc(SourceLocation L) { KeywordLoc = L; }
SourceLocation getEllipsisLoc() const { return EllipsisLoc; }
void setEllipsisLoc(SourceLocation L) { EllipsisLoc = L; }
SourceLocation getColonLoc() const { return ColonLoc; }
void setColonLoc(SourceLocation L) { ColonLoc = L; }

Expr *getLHS() { return reinterpret_cast<Expr*>(SubExprs[LHS]); }
Expr *getRHS() { return reinterpret_cast<Expr*>(SubExprs[RHS]); }
Stmt *getSubStmt() { return SubExprs[SUBSTMT]; }

const Expr *getLHS() const {
return reinterpret_cast<const Expr*>(SubExprs[LHS]);
}

const Expr *getRHS() const {
return reinterpret_cast<const Expr*>(SubExprs[RHS]);
}

const Stmt *getSubStmt() const { return SubExprs[SUBSTMT]; }

void setSubStmt(Stmt *S) { SubExprs[SUBSTMT] = S; }
void setLHS(Expr *Val) { SubExprs[LHS] = reinterpret_cast<Stmt*>(Val); }
void setRHS(Expr *Val) { SubExprs[RHS] = reinterpret_cast<Stmt*>(Val); }

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return KeywordLoc; }

SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY {
const CaseStmt *CS = this;
while (const auto *CS2 = dyn_cast<CaseStmt>(CS->getSubStmt()))
CS = CS2;

return CS->getSubStmt()->getLocEnd();
}

static bool classof(const Stmt *T) {
return T->getStmtClass() == CaseStmtClass;
}

child_range children() {
return child_range(&SubExprs[0], &SubExprs[END_EXPR]);
}
};

class DefaultStmt : public SwitchCase {
Stmt* SubStmt;

public:
DefaultStmt(SourceLocation DL, SourceLocation CL, Stmt *substmt) :
SwitchCase(DefaultStmtClass, DL, CL), SubStmt(substmt) {}

explicit DefaultStmt(EmptyShell Empty)
: SwitchCase(DefaultStmtClass, Empty) {}

Stmt *getSubStmt() { return SubStmt; }
const Stmt *getSubStmt() const { return SubStmt; }
void setSubStmt(Stmt *S) { SubStmt = S; }

SourceLocation getDefaultLoc() const { return KeywordLoc; }
void setDefaultLoc(SourceLocation L) { KeywordLoc = L; }
SourceLocation getColonLoc() const { return ColonLoc; }
void setColonLoc(SourceLocation L) { ColonLoc = L; }

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return KeywordLoc; }
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY {
return SubStmt->getLocEnd();
}

static bool classof(const Stmt *T) {
return T->getStmtClass() == DefaultStmtClass;
}

child_range children() { return child_range(&SubStmt, &SubStmt+1); }
};

inline SourceLocation SwitchCase::getEndLoc() const {
if (const auto *CS = dyn_cast<CaseStmt>(this))
return CS->getLocEnd();
return cast<DefaultStmt>(this)->getLocEnd();
}

class LabelStmt : public Stmt {
SourceLocation IdentLoc;
LabelDecl *TheDecl;
Stmt *SubStmt;

public:
LabelStmt(SourceLocation IL, LabelDecl *D, Stmt *substmt)
: Stmt(LabelStmtClass), IdentLoc(IL), TheDecl(D), SubStmt(substmt) {
static_assert(sizeof(LabelStmt) ==
2 * sizeof(SourceLocation) + 2 * sizeof(void *),
"LabelStmt too big");
}

explicit LabelStmt(EmptyShell Empty) : Stmt(LabelStmtClass, Empty) {}

SourceLocation getIdentLoc() const { return IdentLoc; }
LabelDecl *getDecl() const { return TheDecl; }
void setDecl(LabelDecl *D) { TheDecl = D; }
const char *getName() const;
Stmt *getSubStmt() { return SubStmt; }
const Stmt *getSubStmt() const { return SubStmt; }
void setIdentLoc(SourceLocation L) { IdentLoc = L; }
void setSubStmt(Stmt *SS) { SubStmt = SS; }

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return IdentLoc; }
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY {
return SubStmt->getLocEnd();
}

child_range children() { return child_range(&SubStmt, &SubStmt+1); }

static bool classof(const Stmt *T) {
return T->getStmtClass() == LabelStmtClass;
}
};

class AttributedStmt final
: public Stmt,
private llvm::TrailingObjects<AttributedStmt, const Attr *> {
friend class ASTStmtReader;
friend TrailingObjects;

Stmt *SubStmt;
SourceLocation AttrLoc;
unsigned NumAttrs;

AttributedStmt(SourceLocation Loc, ArrayRef<const Attr*> Attrs, Stmt *SubStmt)
: Stmt(AttributedStmtClass), SubStmt(SubStmt), AttrLoc(Loc),
NumAttrs(Attrs.size()) {
std::copy(Attrs.begin(), Attrs.end(), getAttrArrayPtr());
}

explicit AttributedStmt(EmptyShell Empty, unsigned NumAttrs)
: Stmt(AttributedStmtClass, Empty), NumAttrs(NumAttrs) {
std::fill_n(getAttrArrayPtr(), NumAttrs, nullptr);
}

const Attr *const *getAttrArrayPtr() const {
return getTrailingObjects<const Attr *>();
}
const Attr **getAttrArrayPtr() { return getTrailingObjects<const Attr *>(); }

public:
static AttributedStmt *Create(const ASTContext &C, SourceLocation Loc,
ArrayRef<const Attr*> Attrs, Stmt *SubStmt);

static AttributedStmt *CreateEmpty(const ASTContext &C, unsigned NumAttrs);

SourceLocation getAttrLoc() const { return AttrLoc; }
ArrayRef<const Attr*> getAttrs() const {
return llvm::makeArrayRef(getAttrArrayPtr(), NumAttrs);
}

Stmt *getSubStmt() { return SubStmt; }
const Stmt *getSubStmt() const { return SubStmt; }

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return AttrLoc; }
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY {
return SubStmt->getLocEnd();
}

child_range children() { return child_range(&SubStmt, &SubStmt + 1); }

static bool classof(const Stmt *T) {
return T->getStmtClass() == AttributedStmtClass;
}
};

class IfStmt : public Stmt {
enum { INIT, VAR, COND, THEN, ELSE, END_EXPR };
Stmt* SubExprs[END_EXPR];

SourceLocation IfLoc;
SourceLocation ElseLoc;

public:
IfStmt(const ASTContext &C, SourceLocation IL,
bool IsConstexpr, Stmt *init, VarDecl *var, Expr *cond,
Stmt *then, SourceLocation EL = SourceLocation(),
Stmt *elsev = nullptr);

explicit IfStmt(EmptyShell Empty) : Stmt(IfStmtClass, Empty) {}

VarDecl *getConditionVariable() const;
void setConditionVariable(const ASTContext &C, VarDecl *V);

const DeclStmt *getConditionVariableDeclStmt() const {
return reinterpret_cast<DeclStmt*>(SubExprs[VAR]);
}

Stmt *getInit() { return SubExprs[INIT]; }
const Stmt *getInit() const { return SubExprs[INIT]; }
void setInit(Stmt *S) { SubExprs[INIT] = S; }
const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt *>(E); }
const Stmt *getThen() const { return SubExprs[THEN]; }
void setThen(Stmt *S) { SubExprs[THEN] = S; }
const Stmt *getElse() const { return SubExprs[ELSE]; }
void setElse(Stmt *S) { SubExprs[ELSE] = S; }

Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
Stmt *getThen() { return SubExprs[THEN]; }
Stmt *getElse() { return SubExprs[ELSE]; }

SourceLocation getIfLoc() const { return IfLoc; }
void setIfLoc(SourceLocation L) { IfLoc = L; }
SourceLocation getElseLoc() const { return ElseLoc; }
void setElseLoc(SourceLocation L) { ElseLoc = L; }

bool isConstexpr() const { return IfStmtBits.IsConstexpr; }
void setConstexpr(bool C) { IfStmtBits.IsConstexpr = C; }

bool isObjCAvailabilityCheck() const;

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return IfLoc; }

SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY {
if (SubExprs[ELSE])
return SubExprs[ELSE]->getLocEnd();
else
return SubExprs[THEN]->getLocEnd();
}

child_range children() {
return child_range(&SubExprs[0], &SubExprs[0]+END_EXPR);
}

static bool classof(const Stmt *T) {
return T->getStmtClass() == IfStmtClass;
}
};

class SwitchStmt : public Stmt {
SourceLocation SwitchLoc;
enum { INIT, VAR, COND, BODY, END_EXPR };
Stmt* SubExprs[END_EXPR];

llvm::PointerIntPair<SwitchCase *, 1, bool> FirstCase;

public:
SwitchStmt(const ASTContext &C, Stmt *Init, VarDecl *Var, Expr *cond);

explicit SwitchStmt(EmptyShell Empty) : Stmt(SwitchStmtClass, Empty) {}

VarDecl *getConditionVariable() const;
void setConditionVariable(const ASTContext &C, VarDecl *V);

const DeclStmt *getConditionVariableDeclStmt() const {
return reinterpret_cast<DeclStmt*>(SubExprs[VAR]);
}

Stmt *getInit() { return SubExprs[INIT]; }
const Stmt *getInit() const { return SubExprs[INIT]; }
void setInit(Stmt *S) { SubExprs[INIT] = S; }
const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
const Stmt *getBody() const { return SubExprs[BODY]; }
const SwitchCase *getSwitchCaseList() const { return FirstCase.getPointer(); }

Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]);}
void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt *>(E); }
Stmt *getBody() { return SubExprs[BODY]; }
void setBody(Stmt *S) { SubExprs[BODY] = S; }
SwitchCase *getSwitchCaseList() { return FirstCase.getPointer(); }

void setSwitchCaseList(SwitchCase *SC) { FirstCase.setPointer(SC); }

SourceLocation getSwitchLoc() const { return SwitchLoc; }
void setSwitchLoc(SourceLocation L) { SwitchLoc = L; }

void setBody(Stmt *S, SourceLocation SL) {
SubExprs[BODY] = S;
SwitchLoc = SL;
}

void addSwitchCase(SwitchCase *SC) {
assert(!SC->getNextSwitchCase()
&& "case/default already added to a switch");
SC->setNextSwitchCase(FirstCase.getPointer());
FirstCase.setPointer(SC);
}

void setAllEnumCasesCovered() { FirstCase.setInt(true); }

bool isAllEnumCasesCovered() const { return FirstCase.getInt(); }

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return SwitchLoc; }

SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY {
return SubExprs[BODY] ? SubExprs[BODY]->getLocEnd() : SubExprs[COND]->getLocEnd();
}

child_range children() {
return child_range(&SubExprs[0], &SubExprs[0]+END_EXPR);
}

static bool classof(const Stmt *T) {
return T->getStmtClass() == SwitchStmtClass;
}
};

class WhileStmt : public Stmt {
SourceLocation WhileLoc;
enum { VAR, COND, BODY, END_EXPR };
Stmt* SubExprs[END_EXPR];

public:
WhileStmt(const ASTContext &C, VarDecl *Var, Expr *cond, Stmt *body,
SourceLocation WL);

explicit WhileStmt(EmptyShell Empty) : Stmt(WhileStmtClass, Empty) {}

VarDecl *getConditionVariable() const;
void setConditionVariable(const ASTContext &C, VarDecl *V);

const DeclStmt *getConditionVariableDeclStmt() const {
return reinterpret_cast<DeclStmt*>(SubExprs[VAR]);
}

Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt*>(E); }
Stmt *getBody() { return SubExprs[BODY]; }
const Stmt *getBody() const { return SubExprs[BODY]; }
void setBody(Stmt *S) { SubExprs[BODY] = S; }

SourceLocation getWhileLoc() const { return WhileLoc; }
void setWhileLoc(SourceLocation L) { WhileLoc = L; }

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return WhileLoc; }

SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY {
return SubExprs[BODY]->getLocEnd();
}

static bool classof(const Stmt *T) {
return T->getStmtClass() == WhileStmtClass;
}

child_range children() {
return child_range(&SubExprs[0], &SubExprs[0]+END_EXPR);
}
};

class DoStmt : public Stmt {
SourceLocation DoLoc;
enum { BODY, COND, END_EXPR };
Stmt* SubExprs[END_EXPR];
SourceLocation WhileLoc;
SourceLocation RParenLoc;  

public:
DoStmt(Stmt *body, Expr *cond, SourceLocation DL, SourceLocation WL,
SourceLocation RP)
: Stmt(DoStmtClass), DoLoc(DL), WhileLoc(WL), RParenLoc(RP) {
SubExprs[COND] = reinterpret_cast<Stmt*>(cond);
SubExprs[BODY] = body;
}

explicit DoStmt(EmptyShell Empty) : Stmt(DoStmtClass, Empty) {}

Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt*>(E); }
Stmt *getBody() { return SubExprs[BODY]; }
const Stmt *getBody() const { return SubExprs[BODY]; }
void setBody(Stmt *S) { SubExprs[BODY] = S; }

SourceLocation getDoLoc() const { return DoLoc; }
void setDoLoc(SourceLocation L) { DoLoc = L; }
SourceLocation getWhileLoc() const { return WhileLoc; }
void setWhileLoc(SourceLocation L) { WhileLoc = L; }

SourceLocation getRParenLoc() const { return RParenLoc; }
void setRParenLoc(SourceLocation L) { RParenLoc = L; }

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return DoLoc; }
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY { return RParenLoc; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == DoStmtClass;
}

child_range children() {
return child_range(&SubExprs[0], &SubExprs[0]+END_EXPR);
}
};

class ForStmt : public Stmt {
SourceLocation ForLoc;
enum { INIT, CONDVAR, COND, INC, BODY, END_EXPR };
Stmt* SubExprs[END_EXPR]; 
SourceLocation LParenLoc, RParenLoc;

public:
ForStmt(const ASTContext &C, Stmt *Init, Expr *Cond, VarDecl *condVar,
Expr *Inc, Stmt *Body, SourceLocation FL, SourceLocation LP,
SourceLocation RP);

explicit ForStmt(EmptyShell Empty) : Stmt(ForStmtClass, Empty) {}

Stmt *getInit() { return SubExprs[INIT]; }

VarDecl *getConditionVariable() const;
void setConditionVariable(const ASTContext &C, VarDecl *V);

const DeclStmt *getConditionVariableDeclStmt() const {
return reinterpret_cast<DeclStmt*>(SubExprs[CONDVAR]);
}

Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
Expr *getInc()  { return reinterpret_cast<Expr*>(SubExprs[INC]); }
Stmt *getBody() { return SubExprs[BODY]; }

const Stmt *getInit() const { return SubExprs[INIT]; }
const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
const Expr *getInc()  const { return reinterpret_cast<Expr*>(SubExprs[INC]); }
const Stmt *getBody() const { return SubExprs[BODY]; }

void setInit(Stmt *S) { SubExprs[INIT] = S; }
void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt*>(E); }
void setInc(Expr *E) { SubExprs[INC] = reinterpret_cast<Stmt*>(E); }
void setBody(Stmt *S) { SubExprs[BODY] = S; }

SourceLocation getForLoc() const { return ForLoc; }
void setForLoc(SourceLocation L) { ForLoc = L; }
SourceLocation getLParenLoc() const { return LParenLoc; }
void setLParenLoc(SourceLocation L) { LParenLoc = L; }
SourceLocation getRParenLoc() const { return RParenLoc; }
void setRParenLoc(SourceLocation L) { RParenLoc = L; }

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return ForLoc; }

SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY {
return SubExprs[BODY]->getLocEnd();
}

static bool classof(const Stmt *T) {
return T->getStmtClass() == ForStmtClass;
}

child_range children() {
return child_range(&SubExprs[0], &SubExprs[0]+END_EXPR);
}
};

class GotoStmt : public Stmt {
LabelDecl *Label;
SourceLocation GotoLoc;
SourceLocation LabelLoc;

public:
GotoStmt(LabelDecl *label, SourceLocation GL, SourceLocation LL)
: Stmt(GotoStmtClass), Label(label), GotoLoc(GL), LabelLoc(LL) {}

explicit GotoStmt(EmptyShell Empty) : Stmt(GotoStmtClass, Empty) {}

LabelDecl *getLabel() const { return Label; }
void setLabel(LabelDecl *D) { Label = D; }

SourceLocation getGotoLoc() const { return GotoLoc; }
void setGotoLoc(SourceLocation L) { GotoLoc = L; }
SourceLocation getLabelLoc() const { return LabelLoc; }
void setLabelLoc(SourceLocation L) { LabelLoc = L; }

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return GotoLoc; }
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY { return LabelLoc; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == GotoStmtClass;
}

child_range children() {
return child_range(child_iterator(), child_iterator());
}
};

class IndirectGotoStmt : public Stmt {
SourceLocation GotoLoc;
SourceLocation StarLoc;
Stmt *Target;

public:
IndirectGotoStmt(SourceLocation gotoLoc, SourceLocation starLoc,
Expr *target)
: Stmt(IndirectGotoStmtClass), GotoLoc(gotoLoc), StarLoc(starLoc),
Target((Stmt*)target) {}

explicit IndirectGotoStmt(EmptyShell Empty)
: Stmt(IndirectGotoStmtClass, Empty) {}

void setGotoLoc(SourceLocation L) { GotoLoc = L; }
SourceLocation getGotoLoc() const { return GotoLoc; }
void setStarLoc(SourceLocation L) { StarLoc = L; }
SourceLocation getStarLoc() const { return StarLoc; }

Expr *getTarget() { return reinterpret_cast<Expr*>(Target); }
const Expr *getTarget() const {return reinterpret_cast<const Expr*>(Target);}
void setTarget(Expr *E) { Target = reinterpret_cast<Stmt*>(E); }

LabelDecl *getConstantTarget();
const LabelDecl *getConstantTarget() const {
return const_cast<IndirectGotoStmt*>(this)->getConstantTarget();
}

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return GotoLoc; }
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY { return Target->getLocEnd(); }

static bool classof(const Stmt *T) {
return T->getStmtClass() == IndirectGotoStmtClass;
}

child_range children() { return child_range(&Target, &Target+1); }
};

class ContinueStmt : public Stmt {
SourceLocation ContinueLoc;

public:
ContinueStmt(SourceLocation CL) : Stmt(ContinueStmtClass), ContinueLoc(CL) {}

explicit ContinueStmt(EmptyShell Empty) : Stmt(ContinueStmtClass, Empty) {}

SourceLocation getContinueLoc() const { return ContinueLoc; }
void setContinueLoc(SourceLocation L) { ContinueLoc = L; }

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return ContinueLoc; }
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY { return ContinueLoc; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == ContinueStmtClass;
}

child_range children() {
return child_range(child_iterator(), child_iterator());
}
};

class BreakStmt : public Stmt {
SourceLocation BreakLoc;

public:
BreakStmt(SourceLocation BL) : Stmt(BreakStmtClass), BreakLoc(BL) {
static_assert(sizeof(BreakStmt) == 2 * sizeof(SourceLocation),
"BreakStmt too large");
}

explicit BreakStmt(EmptyShell Empty) : Stmt(BreakStmtClass, Empty) {}

SourceLocation getBreakLoc() const { return BreakLoc; }
void setBreakLoc(SourceLocation L) { BreakLoc = L; }

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return BreakLoc; }
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY { return BreakLoc; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == BreakStmtClass;
}

child_range children() {
return child_range(child_iterator(), child_iterator());
}
};

class ReturnStmt : public Stmt {
SourceLocation RetLoc;
Stmt *RetExpr;
const VarDecl *NRVOCandidate;

public:
explicit ReturnStmt(SourceLocation RL) : ReturnStmt(RL, nullptr, nullptr) {}

ReturnStmt(SourceLocation RL, Expr *E, const VarDecl *NRVOCandidate)
: Stmt(ReturnStmtClass), RetLoc(RL), RetExpr((Stmt *)E),
NRVOCandidate(NRVOCandidate) {}

explicit ReturnStmt(EmptyShell Empty) : Stmt(ReturnStmtClass, Empty) {}

const Expr *getRetValue() const;
Expr *getRetValue();
void setRetValue(Expr *E) { RetExpr = reinterpret_cast<Stmt*>(E); }

SourceLocation getReturnLoc() const { return RetLoc; }
void setReturnLoc(SourceLocation L) { RetLoc = L; }

const VarDecl *getNRVOCandidate() const { return NRVOCandidate; }
void setNRVOCandidate(const VarDecl *Var) { NRVOCandidate = Var; }

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return RetLoc; }

SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY {
return RetExpr ? RetExpr->getLocEnd() : RetLoc;
}

static bool classof(const Stmt *T) {
return T->getStmtClass() == ReturnStmtClass;
}

child_range children() {
if (RetExpr) return child_range(&RetExpr, &RetExpr+1);
return child_range(child_iterator(), child_iterator());
}
};

class AsmStmt : public Stmt {
protected:
friend class ASTStmtReader;

SourceLocation AsmLoc;

bool IsSimple;

bool IsVolatile;

unsigned NumOutputs;
unsigned NumInputs;
unsigned NumClobbers;

Stmt **Exprs = nullptr;

AsmStmt(StmtClass SC, SourceLocation asmloc, bool issimple, bool isvolatile,
unsigned numoutputs, unsigned numinputs, unsigned numclobbers)
: Stmt (SC), AsmLoc(asmloc), IsSimple(issimple), IsVolatile(isvolatile),
NumOutputs(numoutputs), NumInputs(numinputs),
NumClobbers(numclobbers) {}

public:
explicit AsmStmt(StmtClass SC, EmptyShell Empty) : Stmt(SC, Empty) {}

SourceLocation getAsmLoc() const { return AsmLoc; }
void setAsmLoc(SourceLocation L) { AsmLoc = L; }

bool isSimple() const { return IsSimple; }
void setSimple(bool V) { IsSimple = V; }

bool isVolatile() const { return IsVolatile; }
void setVolatile(bool V) { IsVolatile = V; }

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return {}; }
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY { return {}; }


std::string generateAsmString(const ASTContext &C) const;


unsigned getNumOutputs() const { return NumOutputs; }

StringRef getOutputConstraint(unsigned i) const;

bool isOutputPlusConstraint(unsigned i) const {
return getOutputConstraint(i)[0] == '+';
}

const Expr *getOutputExpr(unsigned i) const;

unsigned getNumPlusOperands() const;


unsigned getNumInputs() const { return NumInputs; }

StringRef getInputConstraint(unsigned i) const;

const Expr *getInputExpr(unsigned i) const;


unsigned getNumClobbers() const { return NumClobbers; }
StringRef getClobber(unsigned i) const;

static bool classof(const Stmt *T) {
return T->getStmtClass() == GCCAsmStmtClass ||
T->getStmtClass() == MSAsmStmtClass;
}


using inputs_iterator = ExprIterator;
using const_inputs_iterator = ConstExprIterator;
using inputs_range = llvm::iterator_range<inputs_iterator>;
using inputs_const_range = llvm::iterator_range<const_inputs_iterator>;

inputs_iterator begin_inputs() {
return &Exprs[0] + NumOutputs;
}

inputs_iterator end_inputs() {
return &Exprs[0] + NumOutputs + NumInputs;
}

inputs_range inputs() { return inputs_range(begin_inputs(), end_inputs()); }

const_inputs_iterator begin_inputs() const {
return &Exprs[0] + NumOutputs;
}

const_inputs_iterator end_inputs() const {
return &Exprs[0] + NumOutputs + NumInputs;
}

inputs_const_range inputs() const {
return inputs_const_range(begin_inputs(), end_inputs());
}


using outputs_iterator = ExprIterator;
using const_outputs_iterator = ConstExprIterator;
using outputs_range = llvm::iterator_range<outputs_iterator>;
using outputs_const_range = llvm::iterator_range<const_outputs_iterator>;

outputs_iterator begin_outputs() {
return &Exprs[0];
}

outputs_iterator end_outputs() {
return &Exprs[0] + NumOutputs;
}

outputs_range outputs() {
return outputs_range(begin_outputs(), end_outputs());
}

const_outputs_iterator begin_outputs() const {
return &Exprs[0];
}

const_outputs_iterator end_outputs() const {
return &Exprs[0] + NumOutputs;
}

outputs_const_range outputs() const {
return outputs_const_range(begin_outputs(), end_outputs());
}

child_range children() {
return child_range(&Exprs[0], &Exprs[0] + NumOutputs + NumInputs);
}
};

class GCCAsmStmt : public AsmStmt {
friend class ASTStmtReader;

SourceLocation RParenLoc;
StringLiteral *AsmStr;

StringLiteral **Constraints = nullptr;
StringLiteral **Clobbers = nullptr;
IdentifierInfo **Names = nullptr;

public:
GCCAsmStmt(const ASTContext &C, SourceLocation asmloc, bool issimple,
bool isvolatile, unsigned numoutputs, unsigned numinputs,
IdentifierInfo **names, StringLiteral **constraints, Expr **exprs,
StringLiteral *asmstr, unsigned numclobbers,
StringLiteral **clobbers, SourceLocation rparenloc);

explicit GCCAsmStmt(EmptyShell Empty) : AsmStmt(GCCAsmStmtClass, Empty) {}

SourceLocation getRParenLoc() const { return RParenLoc; }
void setRParenLoc(SourceLocation L) { RParenLoc = L; }


const StringLiteral *getAsmString() const { return AsmStr; }
StringLiteral *getAsmString() { return AsmStr; }
void setAsmString(StringLiteral *E) { AsmStr = E; }

class AsmStringPiece {
public:
enum Kind {
String,  
Operand  
};

private:
Kind MyKind;
std::string Str;
unsigned OperandNo;

CharSourceRange Range;

public:
AsmStringPiece(const std::string &S) : MyKind(String), Str(S) {}
AsmStringPiece(unsigned OpNo, const std::string &S, SourceLocation Begin,
SourceLocation End)
: MyKind(Operand), Str(S), OperandNo(OpNo),
Range(CharSourceRange::getCharRange(Begin, End)) {}

bool isString() const { return MyKind == String; }
bool isOperand() const { return MyKind == Operand; }

const std::string &getString() const { return Str; }

unsigned getOperandNo() const {
assert(isOperand());
return OperandNo;
}

CharSourceRange getRange() const {
assert(isOperand() && "Range is currently used only for Operands.");
return Range;
}

char getModifier() const;
};

unsigned AnalyzeAsmString(SmallVectorImpl<AsmStringPiece> &Pieces,
const ASTContext &C, unsigned &DiagOffs) const;

std::string generateAsmString(const ASTContext &C) const;


IdentifierInfo *getOutputIdentifier(unsigned i) const { return Names[i]; }

StringRef getOutputName(unsigned i) const {
if (IdentifierInfo *II = getOutputIdentifier(i))
return II->getName();

return {};
}

StringRef getOutputConstraint(unsigned i) const;

const StringLiteral *getOutputConstraintLiteral(unsigned i) const {
return Constraints[i];
}
StringLiteral *getOutputConstraintLiteral(unsigned i) {
return Constraints[i];
}

Expr *getOutputExpr(unsigned i);

const Expr *getOutputExpr(unsigned i) const {
return const_cast<GCCAsmStmt*>(this)->getOutputExpr(i);
}


IdentifierInfo *getInputIdentifier(unsigned i) const {
return Names[i + NumOutputs];
}

StringRef getInputName(unsigned i) const {
if (IdentifierInfo *II = getInputIdentifier(i))
return II->getName();

return {};
}

StringRef getInputConstraint(unsigned i) const;

const StringLiteral *getInputConstraintLiteral(unsigned i) const {
return Constraints[i + NumOutputs];
}
StringLiteral *getInputConstraintLiteral(unsigned i) {
return Constraints[i + NumOutputs];
}

Expr *getInputExpr(unsigned i);
void setInputExpr(unsigned i, Expr *E);

const Expr *getInputExpr(unsigned i) const {
return const_cast<GCCAsmStmt*>(this)->getInputExpr(i);
}

private:
void setOutputsAndInputsAndClobbers(const ASTContext &C,
IdentifierInfo **Names,
StringLiteral **Constraints,
Stmt **Exprs,
unsigned NumOutputs,
unsigned NumInputs,
StringLiteral **Clobbers,
unsigned NumClobbers);

public:

int getNamedOperand(StringRef SymbolicName) const;

StringRef getClobber(unsigned i) const;

StringLiteral *getClobberStringLiteral(unsigned i) { return Clobbers[i]; }
const StringLiteral *getClobberStringLiteral(unsigned i) const {
return Clobbers[i];
}

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return AsmLoc; }
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY { return RParenLoc; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == GCCAsmStmtClass;
}
};

class MSAsmStmt : public AsmStmt {
friend class ASTStmtReader;

SourceLocation LBraceLoc, EndLoc;
StringRef AsmStr;

unsigned NumAsmToks = 0;

Token *AsmToks = nullptr;
StringRef *Constraints = nullptr;
StringRef *Clobbers = nullptr;

public:
MSAsmStmt(const ASTContext &C, SourceLocation asmloc,
SourceLocation lbraceloc, bool issimple, bool isvolatile,
ArrayRef<Token> asmtoks, unsigned numoutputs, unsigned numinputs,
ArrayRef<StringRef> constraints,
ArrayRef<Expr*> exprs, StringRef asmstr,
ArrayRef<StringRef> clobbers, SourceLocation endloc);

explicit MSAsmStmt(EmptyShell Empty) : AsmStmt(MSAsmStmtClass, Empty) {}

SourceLocation getLBraceLoc() const { return LBraceLoc; }
void setLBraceLoc(SourceLocation L) { LBraceLoc = L; }
SourceLocation getEndLoc() const { return EndLoc; }
void setEndLoc(SourceLocation L) { EndLoc = L; }

bool hasBraces() const { return LBraceLoc.isValid(); }

unsigned getNumAsmToks() { return NumAsmToks; }
Token *getAsmToks() { return AsmToks; }

StringRef getAsmString() const { return AsmStr; }

std::string generateAsmString(const ASTContext &C) const;


StringRef getOutputConstraint(unsigned i) const {
assert(i < NumOutputs);
return Constraints[i];
}

Expr *getOutputExpr(unsigned i);

const Expr *getOutputExpr(unsigned i) const {
return const_cast<MSAsmStmt*>(this)->getOutputExpr(i);
}


StringRef getInputConstraint(unsigned i) const {
assert(i < NumInputs);
return Constraints[i + NumOutputs];
}

Expr *getInputExpr(unsigned i);
void setInputExpr(unsigned i, Expr *E);

const Expr *getInputExpr(unsigned i) const {
return const_cast<MSAsmStmt*>(this)->getInputExpr(i);
}


ArrayRef<StringRef> getAllConstraints() const {
return llvm::makeArrayRef(Constraints, NumInputs + NumOutputs);
}

ArrayRef<StringRef> getClobbers() const {
return llvm::makeArrayRef(Clobbers, NumClobbers);
}

ArrayRef<Expr*> getAllExprs() const {
return llvm::makeArrayRef(reinterpret_cast<Expr**>(Exprs),
NumInputs + NumOutputs);
}

StringRef getClobber(unsigned i) const { return getClobbers()[i]; }

private:
void initialize(const ASTContext &C, StringRef AsmString,
ArrayRef<Token> AsmToks, ArrayRef<StringRef> Constraints,
ArrayRef<Expr*> Exprs, ArrayRef<StringRef> Clobbers);

public:
SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return AsmLoc; }
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }

static bool classof(const Stmt *T) {
return T->getStmtClass() == MSAsmStmtClass;
}

child_range children() {
return child_range(&Exprs[0], &Exprs[NumInputs + NumOutputs]);
}
};

class SEHExceptStmt : public Stmt {
friend class ASTReader;
friend class ASTStmtReader;

SourceLocation  Loc;
Stmt *Children[2];

enum { FILTER_EXPR, BLOCK };

SEHExceptStmt(SourceLocation Loc, Expr *FilterExpr, Stmt *Block);
explicit SEHExceptStmt(EmptyShell E) : Stmt(SEHExceptStmtClass, E) {}

public:
static SEHExceptStmt* Create(const ASTContext &C,
SourceLocation ExceptLoc,
Expr *FilterExpr,
Stmt *Block);

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return getExceptLoc(); }
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }

SourceLocation getExceptLoc() const { return Loc; }
SourceLocation getEndLoc() const { return getBlock()->getLocEnd(); }

Expr *getFilterExpr() const {
return reinterpret_cast<Expr*>(Children[FILTER_EXPR]);
}

CompoundStmt *getBlock() const {
return cast<CompoundStmt>(Children[BLOCK]);
}

child_range children() {
return child_range(Children, Children+2);
}

static bool classof(const Stmt *T) {
return T->getStmtClass() == SEHExceptStmtClass;
}
};

class SEHFinallyStmt : public Stmt {
friend class ASTReader;
friend class ASTStmtReader;

SourceLocation  Loc;
Stmt *Block;

SEHFinallyStmt(SourceLocation Loc, Stmt *Block);
explicit SEHFinallyStmt(EmptyShell E) : Stmt(SEHFinallyStmtClass, E) {}

public:
static SEHFinallyStmt* Create(const ASTContext &C,
SourceLocation FinallyLoc,
Stmt *Block);

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return getFinallyLoc(); }
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }

SourceLocation getFinallyLoc() const { return Loc; }
SourceLocation getEndLoc() const { return Block->getLocEnd(); }

CompoundStmt *getBlock() const { return cast<CompoundStmt>(Block); }

child_range children() {
return child_range(&Block,&Block+1);
}

static bool classof(const Stmt *T) {
return T->getStmtClass() == SEHFinallyStmtClass;
}
};

class SEHTryStmt : public Stmt {
friend class ASTReader;
friend class ASTStmtReader;

bool IsCXXTry;
SourceLocation  TryLoc;
Stmt *Children[2];

enum { TRY = 0, HANDLER = 1 };

SEHTryStmt(bool isCXXTry, 
SourceLocation TryLoc,
Stmt *TryBlock,
Stmt *Handler);

explicit SEHTryStmt(EmptyShell E) : Stmt(SEHTryStmtClass, E) {}

public:
static SEHTryStmt* Create(const ASTContext &C, bool isCXXTry,
SourceLocation TryLoc, Stmt *TryBlock,
Stmt *Handler);

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return getTryLoc(); }
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }

SourceLocation getTryLoc() const { return TryLoc; }
SourceLocation getEndLoc() const { return Children[HANDLER]->getLocEnd(); }

bool getIsCXXTry() const { return IsCXXTry; }

CompoundStmt* getTryBlock() const {
return cast<CompoundStmt>(Children[TRY]);
}

Stmt *getHandler() const { return Children[HANDLER]; }

SEHExceptStmt  *getExceptHandler() const;
SEHFinallyStmt *getFinallyHandler() const;

child_range children() {
return child_range(Children, Children+2);
}

static bool classof(const Stmt *T) {
return T->getStmtClass() == SEHTryStmtClass;
}
};

class SEHLeaveStmt : public Stmt {
SourceLocation LeaveLoc;

public:
explicit SEHLeaveStmt(SourceLocation LL)
: Stmt(SEHLeaveStmtClass), LeaveLoc(LL) {}

explicit SEHLeaveStmt(EmptyShell Empty) : Stmt(SEHLeaveStmtClass, Empty) {}

SourceLocation getLeaveLoc() const { return LeaveLoc; }
void setLeaveLoc(SourceLocation L) { LeaveLoc = L; }

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return LeaveLoc; }
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY { return LeaveLoc; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == SEHLeaveStmtClass;
}

child_range children() {
return child_range(child_iterator(), child_iterator());
}
};

class CapturedStmt : public Stmt {
public:
enum VariableCaptureKind {
VCK_This,
VCK_ByRef,
VCK_ByCopy,
VCK_VLAType,
};

class Capture {
llvm::PointerIntPair<VarDecl *, 2, VariableCaptureKind> VarAndKind;
SourceLocation Loc;

public:
friend class ASTStmtReader;

Capture(SourceLocation Loc, VariableCaptureKind Kind,
VarDecl *Var = nullptr);

VariableCaptureKind getCaptureKind() const;

SourceLocation getLocation() const { return Loc; }

bool capturesThis() const { return getCaptureKind() == VCK_This; }

bool capturesVariable() const { return getCaptureKind() == VCK_ByRef; }

bool capturesVariableByCopy() const {
return getCaptureKind() == VCK_ByCopy;
}

bool capturesVariableArrayType() const {
return getCaptureKind() == VCK_VLAType;
}

VarDecl *getCapturedVar() const;
};

private:
unsigned NumCaptures;

llvm::PointerIntPair<CapturedDecl *, 2, CapturedRegionKind> CapDeclAndKind;

RecordDecl *TheRecordDecl = nullptr;

CapturedStmt(Stmt *S, CapturedRegionKind Kind, ArrayRef<Capture> Captures,
ArrayRef<Expr *> CaptureInits, CapturedDecl *CD, RecordDecl *RD);

CapturedStmt(EmptyShell Empty, unsigned NumCaptures);

Stmt **getStoredStmts() { return reinterpret_cast<Stmt **>(this + 1); }

Stmt *const *getStoredStmts() const {
return reinterpret_cast<Stmt *const *>(this + 1);
}

Capture *getStoredCaptures() const;

void setCapturedStmt(Stmt *S) { getStoredStmts()[NumCaptures] = S; }

public:
friend class ASTStmtReader;

static CapturedStmt *Create(const ASTContext &Context, Stmt *S,
CapturedRegionKind Kind,
ArrayRef<Capture> Captures,
ArrayRef<Expr *> CaptureInits,
CapturedDecl *CD, RecordDecl *RD);

static CapturedStmt *CreateDeserialized(const ASTContext &Context,
unsigned NumCaptures);

Stmt *getCapturedStmt() { return getStoredStmts()[NumCaptures]; }
const Stmt *getCapturedStmt() const { return getStoredStmts()[NumCaptures]; }

CapturedDecl *getCapturedDecl();
const CapturedDecl *getCapturedDecl() const;

void setCapturedDecl(CapturedDecl *D);

CapturedRegionKind getCapturedRegionKind() const;

void setCapturedRegionKind(CapturedRegionKind Kind);

const RecordDecl *getCapturedRecordDecl() const { return TheRecordDecl; }

void setCapturedRecordDecl(RecordDecl *D) {
assert(D && "null RecordDecl");
TheRecordDecl = D;
}

bool capturesVariable(const VarDecl *Var) const;

using capture_iterator = Capture *;
using const_capture_iterator = const Capture *;
using capture_range = llvm::iterator_range<capture_iterator>;
using capture_const_range = llvm::iterator_range<const_capture_iterator>;

capture_range captures() {
return capture_range(capture_begin(), capture_end());
}
capture_const_range captures() const {
return capture_const_range(capture_begin(), capture_end());
}

capture_iterator capture_begin() { return getStoredCaptures(); }
const_capture_iterator capture_begin() const { return getStoredCaptures(); }

capture_iterator capture_end() const {
return getStoredCaptures() + NumCaptures;
}

unsigned capture_size() const { return NumCaptures; }

using capture_init_iterator = Expr **;
using capture_init_range = llvm::iterator_range<capture_init_iterator>;

using const_capture_init_iterator = Expr *const *;
using const_capture_init_range =
llvm::iterator_range<const_capture_init_iterator>;

capture_init_range capture_inits() {
return capture_init_range(capture_init_begin(), capture_init_end());
}

const_capture_init_range capture_inits() const {
return const_capture_init_range(capture_init_begin(), capture_init_end());
}

capture_init_iterator capture_init_begin() {
return reinterpret_cast<Expr **>(getStoredStmts());
}

const_capture_init_iterator capture_init_begin() const {
return reinterpret_cast<Expr *const *>(getStoredStmts());
}

capture_init_iterator capture_init_end() {
return capture_init_begin() + NumCaptures;
}

const_capture_init_iterator capture_init_end() const {
return capture_init_begin() + NumCaptures;
}

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY {
return getCapturedStmt()->getLocStart();
}

SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY {
return getCapturedStmt()->getLocEnd();
}

SourceRange getSourceRange() const LLVM_READONLY {
return getCapturedStmt()->getSourceRange();
}

static bool classof(const Stmt *T) {
return T->getStmtClass() == CapturedStmtClass;
}

child_range children();
};

} 

#endif 
