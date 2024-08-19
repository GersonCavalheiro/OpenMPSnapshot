
#ifndef LLVM_CLANG_AST_STMTOPENMP_H
#define LLVM_CLANG_AST_STMTOPENMP_H

#include "clang/AST/Expr.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/OpenMPKinds.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {


class OMPExecutableDirective : public Stmt {
friend class ASTStmtReader;
OpenMPDirectiveKind Kind;
SourceLocation StartLoc;
SourceLocation EndLoc;
const unsigned NumClauses;
const unsigned NumChildren;
const unsigned ClausesOffset;

MutableArrayRef<OMPClause *> getClauses() {
OMPClause **ClauseStorage = reinterpret_cast<OMPClause **>(
reinterpret_cast<char *>(this) + ClausesOffset);
return MutableArrayRef<OMPClause *>(ClauseStorage, NumClauses);
}

protected:
template <typename T>
OMPExecutableDirective(const T *, StmtClass SC, OpenMPDirectiveKind K,
SourceLocation StartLoc, SourceLocation EndLoc,
unsigned NumClauses, unsigned NumChildren)
: Stmt(SC), Kind(K), StartLoc(std::move(StartLoc)),
EndLoc(std::move(EndLoc)), NumClauses(NumClauses),
NumChildren(NumChildren),
ClausesOffset(llvm::alignTo(sizeof(T), alignof(OMPClause *))) {}

void setClauses(ArrayRef<OMPClause *> Clauses);

void setAssociatedStmt(Stmt *S) {
assert(hasAssociatedStmt() && "no associated statement.");
*child_begin() = S;
}

public:
template <typename SpecificClause>
class specific_clause_iterator
: public llvm::iterator_adaptor_base<
specific_clause_iterator<SpecificClause>,
ArrayRef<OMPClause *>::const_iterator, std::forward_iterator_tag,
const SpecificClause *, ptrdiff_t, const SpecificClause *,
const SpecificClause *> {
ArrayRef<OMPClause *>::const_iterator End;

void SkipToNextClause() {
while (this->I != End && !isa<SpecificClause>(*this->I))
++this->I;
}

public:
explicit specific_clause_iterator(ArrayRef<OMPClause *> Clauses)
: specific_clause_iterator::iterator_adaptor_base(Clauses.begin()),
End(Clauses.end()) {
SkipToNextClause();
}

const SpecificClause *operator*() const {
return cast<SpecificClause>(*this->I);
}
const SpecificClause *operator->() const { return **this; }

specific_clause_iterator &operator++() {
++this->I;
SkipToNextClause();
return *this;
}
};

template <typename SpecificClause>
static llvm::iterator_range<specific_clause_iterator<SpecificClause>>
getClausesOfKind(ArrayRef<OMPClause *> Clauses) {
return {specific_clause_iterator<SpecificClause>(Clauses),
specific_clause_iterator<SpecificClause>(
llvm::makeArrayRef(Clauses.end(), 0))};
}

template <typename SpecificClause>
llvm::iterator_range<specific_clause_iterator<SpecificClause>>
getClausesOfKind() const {
return getClausesOfKind<SpecificClause>(clauses());
}

template <typename SpecificClause>
const SpecificClause *getSingleClause() const {
auto Clauses = getClausesOfKind<SpecificClause>();

if (Clauses.begin() != Clauses.end()) {
assert(std::next(Clauses.begin()) == Clauses.end() &&
"There are at least 2 clauses of the specified kind");
return *Clauses.begin();
}
return nullptr;
}

template <typename SpecificClause>
bool hasClausesOfKind() const {
auto Clauses = getClausesOfKind<SpecificClause>();
return Clauses.begin() != Clauses.end();
}

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const { return StartLoc; }
SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const { return EndLoc; }

void setLocStart(SourceLocation Loc) { StartLoc = Loc; }
void setLocEnd(SourceLocation Loc) { EndLoc = Loc; }

unsigned getNumClauses() const { return NumClauses; }

OMPClause *getClause(unsigned i) const { return clauses()[i]; }

bool hasAssociatedStmt() const { return NumChildren > 0; }

const Stmt *getAssociatedStmt() const {
assert(hasAssociatedStmt() && "no associated statement.");
return *child_begin();
}
Stmt *getAssociatedStmt() {
assert(hasAssociatedStmt() && "no associated statement.");
return *child_begin();
}

const CapturedStmt *getCapturedStmt(OpenMPDirectiveKind RegionKind) const {
SmallVector<OpenMPDirectiveKind, 4> CaptureRegions;
getOpenMPCaptureRegions(CaptureRegions, getDirectiveKind());
assert(std::any_of(
CaptureRegions.begin(), CaptureRegions.end(),
[=](const OpenMPDirectiveKind K) { return K == RegionKind; }) &&
"RegionKind not found in OpenMP CaptureRegions.");
auto *CS = cast<CapturedStmt>(getAssociatedStmt());
for (auto ThisCaptureRegion : CaptureRegions) {
if (ThisCaptureRegion == RegionKind)
return CS;
CS = cast<CapturedStmt>(CS->getCapturedStmt());
}
llvm_unreachable("Incorrect RegionKind specified for directive.");
}

CapturedStmt *getInnermostCapturedStmt() {
assert(hasAssociatedStmt() && getAssociatedStmt() &&
"Must have associated statement.");
SmallVector<OpenMPDirectiveKind, 4> CaptureRegions;
getOpenMPCaptureRegions(CaptureRegions, getDirectiveKind());
assert(!CaptureRegions.empty() &&
"At least one captured statement must be provided.");
auto *CS = cast<CapturedStmt>(getAssociatedStmt());
for (unsigned Level = CaptureRegions.size(); Level > 1; --Level)
CS = cast<CapturedStmt>(CS->getCapturedStmt());
return CS;
}

const CapturedStmt *getInnermostCapturedStmt() const {
return const_cast<OMPExecutableDirective *>(this)
->getInnermostCapturedStmt();
}

OpenMPDirectiveKind getDirectiveKind() const { return Kind; }

static bool classof(const Stmt *S) {
return S->getStmtClass() >= firstOMPExecutableDirectiveConstant &&
S->getStmtClass() <= lastOMPExecutableDirectiveConstant;
}

child_range children() {
if (!hasAssociatedStmt())
return child_range(child_iterator(), child_iterator());
Stmt **ChildStorage = reinterpret_cast<Stmt **>(getClauses().end());
return child_range(ChildStorage, ChildStorage + 1);
}

ArrayRef<OMPClause *> clauses() { return getClauses(); }

ArrayRef<OMPClause *> clauses() const {
return const_cast<OMPExecutableDirective *>(this)->getClauses();
}
};

class OMPParallelDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
bool HasCancel;

OMPParallelDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned NumClauses)
: OMPExecutableDirective(this, OMPParallelDirectiveClass, OMPD_parallel,
StartLoc, EndLoc, NumClauses, 1),
HasCancel(false) {}

explicit OMPParallelDirective(unsigned NumClauses)
: OMPExecutableDirective(this, OMPParallelDirectiveClass, OMPD_parallel,
SourceLocation(), SourceLocation(), NumClauses,
1),
HasCancel(false) {}

void setHasCancel(bool Has) { HasCancel = Has; }

public:
static OMPParallelDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, bool HasCancel);

static OMPParallelDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses, EmptyShell);

bool hasCancel() const { return HasCancel; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPParallelDirectiveClass;
}
};

class OMPLoopDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
unsigned CollapsedNum;

enum {
AssociatedStmtOffset = 0,
IterationVariableOffset = 1,
LastIterationOffset = 2,
CalcLastIterationOffset = 3,
PreConditionOffset = 4,
CondOffset = 5,
InitOffset = 6,
IncOffset = 7,
PreInitsOffset = 8,
DefaultEnd = 9,
IsLastIterVariableOffset = 9,
LowerBoundVariableOffset = 10,
UpperBoundVariableOffset = 11,
StrideVariableOffset = 12,
EnsureUpperBoundOffset = 13,
NextLowerBoundOffset = 14,
NextUpperBoundOffset = 15,
NumIterationsOffset = 16,
WorksharingEnd = 17,
PrevLowerBoundVariableOffset = 17,
PrevUpperBoundVariableOffset = 18,
DistIncOffset = 19,
PrevEnsureUpperBoundOffset = 20,
CombinedLowerBoundVariableOffset = 21,
CombinedUpperBoundVariableOffset = 22,
CombinedEnsureUpperBoundOffset = 23,
CombinedInitOffset = 24,
CombinedConditionOffset = 25,
CombinedNextLowerBoundOffset = 26,
CombinedNextUpperBoundOffset = 27,
CombinedDistributeEnd = 28,
};

MutableArrayRef<Expr *> getCounters() {
Expr **Storage = reinterpret_cast<Expr **>(
&(*(std::next(child_begin(), getArraysOffset(getDirectiveKind())))));
return MutableArrayRef<Expr *>(Storage, CollapsedNum);
}

MutableArrayRef<Expr *> getPrivateCounters() {
Expr **Storage = reinterpret_cast<Expr **>(&*std::next(
child_begin(), getArraysOffset(getDirectiveKind()) + CollapsedNum));
return MutableArrayRef<Expr *>(Storage, CollapsedNum);
}

MutableArrayRef<Expr *> getInits() {
Expr **Storage = reinterpret_cast<Expr **>(
&*std::next(child_begin(),
getArraysOffset(getDirectiveKind()) + 2 * CollapsedNum));
return MutableArrayRef<Expr *>(Storage, CollapsedNum);
}

MutableArrayRef<Expr *> getUpdates() {
Expr **Storage = reinterpret_cast<Expr **>(
&*std::next(child_begin(),
getArraysOffset(getDirectiveKind()) + 3 * CollapsedNum));
return MutableArrayRef<Expr *>(Storage, CollapsedNum);
}

MutableArrayRef<Expr *> getFinals() {
Expr **Storage = reinterpret_cast<Expr **>(
&*std::next(child_begin(),
getArraysOffset(getDirectiveKind()) + 4 * CollapsedNum));
return MutableArrayRef<Expr *>(Storage, CollapsedNum);
}

protected:
template <typename T>
OMPLoopDirective(const T *That, StmtClass SC, OpenMPDirectiveKind Kind,
SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, unsigned NumClauses,
unsigned NumSpecialChildren = 0)
: OMPExecutableDirective(That, SC, Kind, StartLoc, EndLoc, NumClauses,
numLoopChildren(CollapsedNum, Kind) +
NumSpecialChildren),
CollapsedNum(CollapsedNum) {}

static unsigned getArraysOffset(OpenMPDirectiveKind Kind) {
if (isOpenMPLoopBoundSharingDirective(Kind))
return CombinedDistributeEnd;
if (isOpenMPWorksharingDirective(Kind) || isOpenMPTaskLoopDirective(Kind) ||
isOpenMPDistributeDirective(Kind))
return WorksharingEnd;
return DefaultEnd;
}

static unsigned numLoopChildren(unsigned CollapsedNum,
OpenMPDirectiveKind Kind) {
return getArraysOffset(Kind) + 5 * CollapsedNum; 
}

void setIterationVariable(Expr *IV) {
*std::next(child_begin(), IterationVariableOffset) = IV;
}
void setLastIteration(Expr *LI) {
*std::next(child_begin(), LastIterationOffset) = LI;
}
void setCalcLastIteration(Expr *CLI) {
*std::next(child_begin(), CalcLastIterationOffset) = CLI;
}
void setPreCond(Expr *PC) {
*std::next(child_begin(), PreConditionOffset) = PC;
}
void setCond(Expr *Cond) {
*std::next(child_begin(), CondOffset) = Cond;
}
void setInit(Expr *Init) { *std::next(child_begin(), InitOffset) = Init; }
void setInc(Expr *Inc) { *std::next(child_begin(), IncOffset) = Inc; }
void setPreInits(Stmt *PreInits) {
*std::next(child_begin(), PreInitsOffset) = PreInits;
}
void setIsLastIterVariable(Expr *IL) {
assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
isOpenMPTaskLoopDirective(getDirectiveKind()) ||
isOpenMPDistributeDirective(getDirectiveKind())) &&
"expected worksharing loop directive");
*std::next(child_begin(), IsLastIterVariableOffset) = IL;
}
void setLowerBoundVariable(Expr *LB) {
assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
isOpenMPTaskLoopDirective(getDirectiveKind()) ||
isOpenMPDistributeDirective(getDirectiveKind())) &&
"expected worksharing loop directive");
*std::next(child_begin(), LowerBoundVariableOffset) = LB;
}
void setUpperBoundVariable(Expr *UB) {
assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
isOpenMPTaskLoopDirective(getDirectiveKind()) ||
isOpenMPDistributeDirective(getDirectiveKind())) &&
"expected worksharing loop directive");
*std::next(child_begin(), UpperBoundVariableOffset) = UB;
}
void setStrideVariable(Expr *ST) {
assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
isOpenMPTaskLoopDirective(getDirectiveKind()) ||
isOpenMPDistributeDirective(getDirectiveKind())) &&
"expected worksharing loop directive");
*std::next(child_begin(), StrideVariableOffset) = ST;
}
void setEnsureUpperBound(Expr *EUB) {
assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
isOpenMPTaskLoopDirective(getDirectiveKind()) ||
isOpenMPDistributeDirective(getDirectiveKind())) &&
"expected worksharing loop directive");
*std::next(child_begin(), EnsureUpperBoundOffset) = EUB;
}
void setNextLowerBound(Expr *NLB) {
assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
isOpenMPTaskLoopDirective(getDirectiveKind()) ||
isOpenMPDistributeDirective(getDirectiveKind())) &&
"expected worksharing loop directive");
*std::next(child_begin(), NextLowerBoundOffset) = NLB;
}
void setNextUpperBound(Expr *NUB) {
assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
isOpenMPTaskLoopDirective(getDirectiveKind()) ||
isOpenMPDistributeDirective(getDirectiveKind())) &&
"expected worksharing loop directive");
*std::next(child_begin(), NextUpperBoundOffset) = NUB;
}
void setNumIterations(Expr *NI) {
assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
isOpenMPTaskLoopDirective(getDirectiveKind()) ||
isOpenMPDistributeDirective(getDirectiveKind())) &&
"expected worksharing loop directive");
*std::next(child_begin(), NumIterationsOffset) = NI;
}
void setPrevLowerBoundVariable(Expr *PrevLB) {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
*std::next(child_begin(), PrevLowerBoundVariableOffset) = PrevLB;
}
void setPrevUpperBoundVariable(Expr *PrevUB) {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
*std::next(child_begin(), PrevUpperBoundVariableOffset) = PrevUB;
}
void setDistInc(Expr *DistInc) {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
*std::next(child_begin(), DistIncOffset) = DistInc;
}
void setPrevEnsureUpperBound(Expr *PrevEUB) {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
*std::next(child_begin(), PrevEnsureUpperBoundOffset) = PrevEUB;
}
void setCombinedLowerBoundVariable(Expr *CombLB) {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
*std::next(child_begin(), CombinedLowerBoundVariableOffset) = CombLB;
}
void setCombinedUpperBoundVariable(Expr *CombUB) {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
*std::next(child_begin(), CombinedUpperBoundVariableOffset) = CombUB;
}
void setCombinedEnsureUpperBound(Expr *CombEUB) {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
*std::next(child_begin(), CombinedEnsureUpperBoundOffset) = CombEUB;
}
void setCombinedInit(Expr *CombInit) {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
*std::next(child_begin(), CombinedInitOffset) = CombInit;
}
void setCombinedCond(Expr *CombCond) {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
*std::next(child_begin(), CombinedConditionOffset) = CombCond;
}
void setCombinedNextLowerBound(Expr *CombNLB) {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
*std::next(child_begin(), CombinedNextLowerBoundOffset) = CombNLB;
}
void setCombinedNextUpperBound(Expr *CombNUB) {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
*std::next(child_begin(), CombinedNextUpperBoundOffset) = CombNUB;
}
void setCounters(ArrayRef<Expr *> A);
void setPrivateCounters(ArrayRef<Expr *> A);
void setInits(ArrayRef<Expr *> A);
void setUpdates(ArrayRef<Expr *> A);
void setFinals(ArrayRef<Expr *> A);

public:
struct DistCombinedHelperExprs {
Expr *LB;
Expr *UB;
Expr *EUB;
Expr *Init;
Expr *Cond;
Expr *NLB;
Expr *NUB;
};

struct HelperExprs {
Expr *IterationVarRef;
Expr *LastIteration;
Expr *NumIterations;
Expr *CalcLastIteration;
Expr *PreCond;
Expr *Cond;
Expr *Init;
Expr *Inc;
Expr *IL;
Expr *LB;
Expr *UB;
Expr *ST;
Expr *EUB;
Expr *NLB;
Expr *NUB;
Expr *PrevLB;
Expr *PrevUB;
Expr *DistInc;
Expr *PrevEUB;
SmallVector<Expr *, 4> Counters;
SmallVector<Expr *, 4> PrivateCounters;
SmallVector<Expr *, 4> Inits;
SmallVector<Expr *, 4> Updates;
SmallVector<Expr *, 4> Finals;
Stmt *PreInits;

DistCombinedHelperExprs DistCombinedFields;

bool builtAll() {
return IterationVarRef != nullptr && LastIteration != nullptr &&
NumIterations != nullptr && PreCond != nullptr &&
Cond != nullptr && Init != nullptr && Inc != nullptr;
}

void clear(unsigned Size) {
IterationVarRef = nullptr;
LastIteration = nullptr;
CalcLastIteration = nullptr;
PreCond = nullptr;
Cond = nullptr;
Init = nullptr;
Inc = nullptr;
IL = nullptr;
LB = nullptr;
UB = nullptr;
ST = nullptr;
EUB = nullptr;
NLB = nullptr;
NUB = nullptr;
NumIterations = nullptr;
PrevLB = nullptr;
PrevUB = nullptr;
DistInc = nullptr;
PrevEUB = nullptr;
Counters.resize(Size);
PrivateCounters.resize(Size);
Inits.resize(Size);
Updates.resize(Size);
Finals.resize(Size);
for (unsigned i = 0; i < Size; ++i) {
Counters[i] = nullptr;
PrivateCounters[i] = nullptr;
Inits[i] = nullptr;
Updates[i] = nullptr;
Finals[i] = nullptr;
}
PreInits = nullptr;
DistCombinedFields.LB = nullptr;
DistCombinedFields.UB = nullptr;
DistCombinedFields.EUB = nullptr;
DistCombinedFields.Init = nullptr;
DistCombinedFields.Cond = nullptr;
DistCombinedFields.NLB = nullptr;
DistCombinedFields.NUB = nullptr;
}
};

unsigned getCollapsedNumber() const { return CollapsedNum; }

Expr *getIterationVariable() const {
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), IterationVariableOffset)));
}
Expr *getLastIteration() const {
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), LastIterationOffset)));
}
Expr *getCalcLastIteration() const {
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), CalcLastIterationOffset)));
}
Expr *getPreCond() const {
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), PreConditionOffset)));
}
Expr *getCond() const {
return const_cast<Expr *>(
reinterpret_cast<const Expr *>(*std::next(child_begin(), CondOffset)));
}
Expr *getInit() const {
return const_cast<Expr *>(
reinterpret_cast<const Expr *>(*std::next(child_begin(), InitOffset)));
}
Expr *getInc() const {
return const_cast<Expr *>(
reinterpret_cast<const Expr *>(*std::next(child_begin(), IncOffset)));
}
const Stmt *getPreInits() const {
return *std::next(child_begin(), PreInitsOffset);
}
Stmt *getPreInits() { return *std::next(child_begin(), PreInitsOffset); }
Expr *getIsLastIterVariable() const {
assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
isOpenMPTaskLoopDirective(getDirectiveKind()) ||
isOpenMPDistributeDirective(getDirectiveKind())) &&
"expected worksharing loop directive");
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), IsLastIterVariableOffset)));
}
Expr *getLowerBoundVariable() const {
assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
isOpenMPTaskLoopDirective(getDirectiveKind()) ||
isOpenMPDistributeDirective(getDirectiveKind())) &&
"expected worksharing loop directive");
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), LowerBoundVariableOffset)));
}
Expr *getUpperBoundVariable() const {
assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
isOpenMPTaskLoopDirective(getDirectiveKind()) ||
isOpenMPDistributeDirective(getDirectiveKind())) &&
"expected worksharing loop directive");
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), UpperBoundVariableOffset)));
}
Expr *getStrideVariable() const {
assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
isOpenMPTaskLoopDirective(getDirectiveKind()) ||
isOpenMPDistributeDirective(getDirectiveKind())) &&
"expected worksharing loop directive");
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), StrideVariableOffset)));
}
Expr *getEnsureUpperBound() const {
assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
isOpenMPTaskLoopDirective(getDirectiveKind()) ||
isOpenMPDistributeDirective(getDirectiveKind())) &&
"expected worksharing loop directive");
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), EnsureUpperBoundOffset)));
}
Expr *getNextLowerBound() const {
assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
isOpenMPTaskLoopDirective(getDirectiveKind()) ||
isOpenMPDistributeDirective(getDirectiveKind())) &&
"expected worksharing loop directive");
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), NextLowerBoundOffset)));
}
Expr *getNextUpperBound() const {
assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
isOpenMPTaskLoopDirective(getDirectiveKind()) ||
isOpenMPDistributeDirective(getDirectiveKind())) &&
"expected worksharing loop directive");
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), NextUpperBoundOffset)));
}
Expr *getNumIterations() const {
assert((isOpenMPWorksharingDirective(getDirectiveKind()) ||
isOpenMPTaskLoopDirective(getDirectiveKind()) ||
isOpenMPDistributeDirective(getDirectiveKind())) &&
"expected worksharing loop directive");
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), NumIterationsOffset)));
}
Expr *getPrevLowerBoundVariable() const {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), PrevLowerBoundVariableOffset)));
}
Expr *getPrevUpperBoundVariable() const {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), PrevUpperBoundVariableOffset)));
}
Expr *getDistInc() const {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), DistIncOffset)));
}
Expr *getPrevEnsureUpperBound() const {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), PrevEnsureUpperBoundOffset)));
}
Expr *getCombinedLowerBoundVariable() const {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), CombinedLowerBoundVariableOffset)));
}
Expr *getCombinedUpperBoundVariable() const {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), CombinedUpperBoundVariableOffset)));
}
Expr *getCombinedEnsureUpperBound() const {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), CombinedEnsureUpperBoundOffset)));
}
Expr *getCombinedInit() const {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), CombinedInitOffset)));
}
Expr *getCombinedCond() const {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), CombinedConditionOffset)));
}
Expr *getCombinedNextLowerBound() const {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), CombinedNextLowerBoundOffset)));
}
Expr *getCombinedNextUpperBound() const {
assert(isOpenMPLoopBoundSharingDirective(getDirectiveKind()) &&
"expected loop bound sharing directive");
return const_cast<Expr *>(reinterpret_cast<const Expr *>(
*std::next(child_begin(), CombinedNextUpperBoundOffset)));
}
const Stmt *getBody() const {
const Stmt *Body =
getInnermostCapturedStmt()->getCapturedStmt()->IgnoreContainers();
Body = cast<ForStmt>(Body)->getBody();
for (unsigned Cnt = 1; Cnt < CollapsedNum; ++Cnt) {
Body = Body->IgnoreContainers();
Body = cast<ForStmt>(Body)->getBody();
}
return Body;
}

ArrayRef<Expr *> counters() { return getCounters(); }

ArrayRef<Expr *> counters() const {
return const_cast<OMPLoopDirective *>(this)->getCounters();
}

ArrayRef<Expr *> private_counters() { return getPrivateCounters(); }

ArrayRef<Expr *> private_counters() const {
return const_cast<OMPLoopDirective *>(this)->getPrivateCounters();
}

ArrayRef<Expr *> inits() { return getInits(); }

ArrayRef<Expr *> inits() const {
return const_cast<OMPLoopDirective *>(this)->getInits();
}

ArrayRef<Expr *> updates() { return getUpdates(); }

ArrayRef<Expr *> updates() const {
return const_cast<OMPLoopDirective *>(this)->getUpdates();
}

ArrayRef<Expr *> finals() { return getFinals(); }

ArrayRef<Expr *> finals() const {
return const_cast<OMPLoopDirective *>(this)->getFinals();
}

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPSimdDirectiveClass ||
T->getStmtClass() == OMPForDirectiveClass ||
T->getStmtClass() == OMPForSimdDirectiveClass ||
T->getStmtClass() == OMPParallelForDirectiveClass ||
T->getStmtClass() == OMPParallelForSimdDirectiveClass ||
T->getStmtClass() == OMPTaskLoopDirectiveClass ||
T->getStmtClass() == OMPTaskLoopSimdDirectiveClass ||
T->getStmtClass() == OMPDistributeDirectiveClass ||
T->getStmtClass() == OMPTargetParallelForDirectiveClass ||
T->getStmtClass() == OMPDistributeParallelForDirectiveClass ||
T->getStmtClass() == OMPDistributeParallelForSimdDirectiveClass ||
T->getStmtClass() == OMPDistributeSimdDirectiveClass ||
T->getStmtClass() == OMPTargetParallelForSimdDirectiveClass ||
T->getStmtClass() == OMPTargetSimdDirectiveClass ||
T->getStmtClass() == OMPTeamsDistributeDirectiveClass ||
T->getStmtClass() == OMPTeamsDistributeSimdDirectiveClass ||
T->getStmtClass() ==
OMPTeamsDistributeParallelForSimdDirectiveClass ||
T->getStmtClass() == OMPTeamsDistributeParallelForDirectiveClass ||
T->getStmtClass() ==
OMPTargetTeamsDistributeParallelForDirectiveClass ||
T->getStmtClass() ==
OMPTargetTeamsDistributeParallelForSimdDirectiveClass ||
T->getStmtClass() == OMPTargetTeamsDistributeDirectiveClass ||
T->getStmtClass() == OMPTargetTeamsDistributeSimdDirectiveClass;
}
};

class OMPSimdDirective : public OMPLoopDirective {
friend class ASTStmtReader;
OMPSimdDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPSimdDirectiveClass, OMPD_simd, StartLoc,
EndLoc, CollapsedNum, NumClauses) {}

explicit OMPSimdDirective(unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPSimdDirectiveClass, OMPD_simd,
SourceLocation(), SourceLocation(), CollapsedNum,
NumClauses) {}

public:
static OMPSimdDirective *Create(const ASTContext &C, SourceLocation StartLoc,
SourceLocation EndLoc, unsigned CollapsedNum,
ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt,
const HelperExprs &Exprs);

static OMPSimdDirective *CreateEmpty(const ASTContext &C, unsigned NumClauses,
unsigned CollapsedNum, EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPSimdDirectiveClass;
}
};

class OMPForDirective : public OMPLoopDirective {
friend class ASTStmtReader;

bool HasCancel;

OMPForDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPForDirectiveClass, OMPD_for, StartLoc, EndLoc,
CollapsedNum, NumClauses),
HasCancel(false) {}

explicit OMPForDirective(unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPForDirectiveClass, OMPD_for, SourceLocation(),
SourceLocation(), CollapsedNum, NumClauses),
HasCancel(false) {}

void setHasCancel(bool Has) { HasCancel = Has; }

public:
static OMPForDirective *Create(const ASTContext &C, SourceLocation StartLoc,
SourceLocation EndLoc, unsigned CollapsedNum,
ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs,
bool HasCancel);

static OMPForDirective *CreateEmpty(const ASTContext &C, unsigned NumClauses,
unsigned CollapsedNum, EmptyShell);

bool hasCancel() const { return HasCancel; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPForDirectiveClass;
}
};

class OMPForSimdDirective : public OMPLoopDirective {
friend class ASTStmtReader;
OMPForSimdDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPForSimdDirectiveClass, OMPD_for_simd,
StartLoc, EndLoc, CollapsedNum, NumClauses) {}

explicit OMPForSimdDirective(unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPForSimdDirectiveClass, OMPD_for_simd,
SourceLocation(), SourceLocation(), CollapsedNum,
NumClauses) {}

public:
static OMPForSimdDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs);

static OMPForSimdDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses,
unsigned CollapsedNum, EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPForSimdDirectiveClass;
}
};

class OMPSectionsDirective : public OMPExecutableDirective {
friend class ASTStmtReader;

bool HasCancel;

OMPSectionsDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned NumClauses)
: OMPExecutableDirective(this, OMPSectionsDirectiveClass, OMPD_sections,
StartLoc, EndLoc, NumClauses, 1),
HasCancel(false) {}

explicit OMPSectionsDirective(unsigned NumClauses)
: OMPExecutableDirective(this, OMPSectionsDirectiveClass, OMPD_sections,
SourceLocation(), SourceLocation(), NumClauses,
1),
HasCancel(false) {}

void setHasCancel(bool Has) { HasCancel = Has; }

public:
static OMPSectionsDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, bool HasCancel);

static OMPSectionsDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses, EmptyShell);

bool hasCancel() const { return HasCancel; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPSectionsDirectiveClass;
}
};

class OMPSectionDirective : public OMPExecutableDirective {
friend class ASTStmtReader;

bool HasCancel;

OMPSectionDirective(SourceLocation StartLoc, SourceLocation EndLoc)
: OMPExecutableDirective(this, OMPSectionDirectiveClass, OMPD_section,
StartLoc, EndLoc, 0, 1),
HasCancel(false) {}

explicit OMPSectionDirective()
: OMPExecutableDirective(this, OMPSectionDirectiveClass, OMPD_section,
SourceLocation(), SourceLocation(), 0, 1),
HasCancel(false) {}

public:
static OMPSectionDirective *Create(const ASTContext &C,
SourceLocation StartLoc,
SourceLocation EndLoc,
Stmt *AssociatedStmt, bool HasCancel);

static OMPSectionDirective *CreateEmpty(const ASTContext &C, EmptyShell);

void setHasCancel(bool Has) { HasCancel = Has; }

bool hasCancel() const { return HasCancel; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPSectionDirectiveClass;
}
};

class OMPSingleDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
OMPSingleDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned NumClauses)
: OMPExecutableDirective(this, OMPSingleDirectiveClass, OMPD_single,
StartLoc, EndLoc, NumClauses, 1) {}

explicit OMPSingleDirective(unsigned NumClauses)
: OMPExecutableDirective(this, OMPSingleDirectiveClass, OMPD_single,
SourceLocation(), SourceLocation(), NumClauses,
1) {}

public:
static OMPSingleDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

static OMPSingleDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses, EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPSingleDirectiveClass;
}
};

class OMPMasterDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
OMPMasterDirective(SourceLocation StartLoc, SourceLocation EndLoc)
: OMPExecutableDirective(this, OMPMasterDirectiveClass, OMPD_master,
StartLoc, EndLoc, 0, 1) {}

explicit OMPMasterDirective()
: OMPExecutableDirective(this, OMPMasterDirectiveClass, OMPD_master,
SourceLocation(), SourceLocation(), 0, 1) {}

public:
static OMPMasterDirective *Create(const ASTContext &C,
SourceLocation StartLoc,
SourceLocation EndLoc,
Stmt *AssociatedStmt);

static OMPMasterDirective *CreateEmpty(const ASTContext &C, EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPMasterDirectiveClass;
}
};

class OMPCriticalDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
DeclarationNameInfo DirName;
OMPCriticalDirective(const DeclarationNameInfo &Name, SourceLocation StartLoc,
SourceLocation EndLoc, unsigned NumClauses)
: OMPExecutableDirective(this, OMPCriticalDirectiveClass, OMPD_critical,
StartLoc, EndLoc, NumClauses, 1),
DirName(Name) {}

explicit OMPCriticalDirective(unsigned NumClauses)
: OMPExecutableDirective(this, OMPCriticalDirectiveClass, OMPD_critical,
SourceLocation(), SourceLocation(), NumClauses,
1),
DirName() {}

void setDirectiveName(const DeclarationNameInfo &Name) { DirName = Name; }

public:
static OMPCriticalDirective *
Create(const ASTContext &C, const DeclarationNameInfo &Name,
SourceLocation StartLoc, SourceLocation EndLoc,
ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

static OMPCriticalDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses, EmptyShell);

DeclarationNameInfo getDirectiveName() const { return DirName; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPCriticalDirectiveClass;
}
};

class OMPParallelForDirective : public OMPLoopDirective {
friend class ASTStmtReader;

bool HasCancel;

OMPParallelForDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPParallelForDirectiveClass, OMPD_parallel_for,
StartLoc, EndLoc, CollapsedNum, NumClauses),
HasCancel(false) {}

explicit OMPParallelForDirective(unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPParallelForDirectiveClass, OMPD_parallel_for,
SourceLocation(), SourceLocation(), CollapsedNum,
NumClauses),
HasCancel(false) {}

void setHasCancel(bool Has) { HasCancel = Has; }

public:
static OMPParallelForDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs, bool HasCancel);

static OMPParallelForDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses,
unsigned CollapsedNum,
EmptyShell);

bool hasCancel() const { return HasCancel; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPParallelForDirectiveClass;
}
};

class OMPParallelForSimdDirective : public OMPLoopDirective {
friend class ASTStmtReader;
OMPParallelForSimdDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPParallelForSimdDirectiveClass,
OMPD_parallel_for_simd, StartLoc, EndLoc, CollapsedNum,
NumClauses) {}

explicit OMPParallelForSimdDirective(unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(this, OMPParallelForSimdDirectiveClass,
OMPD_parallel_for_simd, SourceLocation(),
SourceLocation(), CollapsedNum, NumClauses) {}

public:
static OMPParallelForSimdDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs);

static OMPParallelForSimdDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses,
unsigned CollapsedNum,
EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPParallelForSimdDirectiveClass;
}
};

class OMPParallelSectionsDirective : public OMPExecutableDirective {
friend class ASTStmtReader;

bool HasCancel;

OMPParallelSectionsDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned NumClauses)
: OMPExecutableDirective(this, OMPParallelSectionsDirectiveClass,
OMPD_parallel_sections, StartLoc, EndLoc,
NumClauses, 1),
HasCancel(false) {}

explicit OMPParallelSectionsDirective(unsigned NumClauses)
: OMPExecutableDirective(this, OMPParallelSectionsDirectiveClass,
OMPD_parallel_sections, SourceLocation(),
SourceLocation(), NumClauses, 1),
HasCancel(false) {}

void setHasCancel(bool Has) { HasCancel = Has; }

public:
static OMPParallelSectionsDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, bool HasCancel);

static OMPParallelSectionsDirective *
CreateEmpty(const ASTContext &C, unsigned NumClauses, EmptyShell);

bool hasCancel() const { return HasCancel; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPParallelSectionsDirectiveClass;
}
};

class OMPTaskDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
bool HasCancel;

OMPTaskDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned NumClauses)
: OMPExecutableDirective(this, OMPTaskDirectiveClass, OMPD_task, StartLoc,
EndLoc, NumClauses, 1),
HasCancel(false) {}

explicit OMPTaskDirective(unsigned NumClauses)
: OMPExecutableDirective(this, OMPTaskDirectiveClass, OMPD_task,
SourceLocation(), SourceLocation(), NumClauses,
1),
HasCancel(false) {}

void setHasCancel(bool Has) { HasCancel = Has; }

public:
static OMPTaskDirective *Create(const ASTContext &C, SourceLocation StartLoc,
SourceLocation EndLoc,
ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, bool HasCancel);

static OMPTaskDirective *CreateEmpty(const ASTContext &C, unsigned NumClauses,
EmptyShell);

bool hasCancel() const { return HasCancel; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTaskDirectiveClass;
}
};

class OMPTaskyieldDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
OMPTaskyieldDirective(SourceLocation StartLoc, SourceLocation EndLoc)
: OMPExecutableDirective(this, OMPTaskyieldDirectiveClass, OMPD_taskyield,
StartLoc, EndLoc, 0, 0) {}

explicit OMPTaskyieldDirective()
: OMPExecutableDirective(this, OMPTaskyieldDirectiveClass, OMPD_taskyield,
SourceLocation(), SourceLocation(), 0, 0) {}

public:
static OMPTaskyieldDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc);

static OMPTaskyieldDirective *CreateEmpty(const ASTContext &C, EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTaskyieldDirectiveClass;
}
};

class OMPBarrierDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
OMPBarrierDirective(SourceLocation StartLoc, SourceLocation EndLoc)
: OMPExecutableDirective(this, OMPBarrierDirectiveClass, OMPD_barrier,
StartLoc, EndLoc, 0, 0) {}

explicit OMPBarrierDirective()
: OMPExecutableDirective(this, OMPBarrierDirectiveClass, OMPD_barrier,
SourceLocation(), SourceLocation(), 0, 0) {}

public:
static OMPBarrierDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc);

static OMPBarrierDirective *CreateEmpty(const ASTContext &C, EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPBarrierDirectiveClass;
}
};

class OMPTaskwaitDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
OMPTaskwaitDirective(SourceLocation StartLoc, SourceLocation EndLoc)
: OMPExecutableDirective(this, OMPTaskwaitDirectiveClass, OMPD_taskwait,
StartLoc, EndLoc, 0, 0) {}

explicit OMPTaskwaitDirective()
: OMPExecutableDirective(this, OMPTaskwaitDirectiveClass, OMPD_taskwait,
SourceLocation(), SourceLocation(), 0, 0) {}

public:
static OMPTaskwaitDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc);

static OMPTaskwaitDirective *CreateEmpty(const ASTContext &C, EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTaskwaitDirectiveClass;
}
};

class OMPTaskgroupDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
OMPTaskgroupDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned NumClauses)
: OMPExecutableDirective(this, OMPTaskgroupDirectiveClass, OMPD_taskgroup,
StartLoc, EndLoc, NumClauses, 2) {}

explicit OMPTaskgroupDirective(unsigned NumClauses)
: OMPExecutableDirective(this, OMPTaskgroupDirectiveClass, OMPD_taskgroup,
SourceLocation(), SourceLocation(), NumClauses,
2) {}

void setReductionRef(Expr *RR) {
*std::next(child_begin(), 1) = RR;
}

public:
static OMPTaskgroupDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt,
Expr *ReductionRef);

static OMPTaskgroupDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses, EmptyShell);


const Expr *getReductionRef() const {
return static_cast<const Expr *>(*std::next(child_begin(), 1));
}
Expr *getReductionRef() {
return static_cast<Expr *>(*std::next(child_begin(), 1));
}

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTaskgroupDirectiveClass;
}
};

class OMPFlushDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
OMPFlushDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned NumClauses)
: OMPExecutableDirective(this, OMPFlushDirectiveClass, OMPD_flush,
StartLoc, EndLoc, NumClauses, 0) {}

explicit OMPFlushDirective(unsigned NumClauses)
: OMPExecutableDirective(this, OMPFlushDirectiveClass, OMPD_flush,
SourceLocation(), SourceLocation(), NumClauses,
0) {}

public:
static OMPFlushDirective *Create(const ASTContext &C, SourceLocation StartLoc,
SourceLocation EndLoc,
ArrayRef<OMPClause *> Clauses);

static OMPFlushDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses, EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPFlushDirectiveClass;
}
};

class OMPOrderedDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
OMPOrderedDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned NumClauses)
: OMPExecutableDirective(this, OMPOrderedDirectiveClass, OMPD_ordered,
StartLoc, EndLoc, NumClauses, 1) {}

explicit OMPOrderedDirective(unsigned NumClauses)
: OMPExecutableDirective(this, OMPOrderedDirectiveClass, OMPD_ordered,
SourceLocation(), SourceLocation(), NumClauses,
1) {}

public:
static OMPOrderedDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

static OMPOrderedDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses, EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPOrderedDirectiveClass;
}
};

class OMPAtomicDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
bool IsXLHSInRHSPart;
bool IsPostfixUpdate;

OMPAtomicDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned NumClauses)
: OMPExecutableDirective(this, OMPAtomicDirectiveClass, OMPD_atomic,
StartLoc, EndLoc, NumClauses, 5),
IsXLHSInRHSPart(false), IsPostfixUpdate(false) {}

explicit OMPAtomicDirective(unsigned NumClauses)
: OMPExecutableDirective(this, OMPAtomicDirectiveClass, OMPD_atomic,
SourceLocation(), SourceLocation(), NumClauses,
5),
IsXLHSInRHSPart(false), IsPostfixUpdate(false) {}

void setX(Expr *X) { *std::next(child_begin()) = X; }
void setUpdateExpr(Expr *UE) { *std::next(child_begin(), 2) = UE; }
void setV(Expr *V) { *std::next(child_begin(), 3) = V; }
void setExpr(Expr *E) { *std::next(child_begin(), 4) = E; }

public:
static OMPAtomicDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *X, Expr *V,
Expr *E, Expr *UE, bool IsXLHSInRHSPart, bool IsPostfixUpdate);

static OMPAtomicDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses, EmptyShell);

Expr *getX() { return cast_or_null<Expr>(*std::next(child_begin())); }
const Expr *getX() const {
return cast_or_null<Expr>(*std::next(child_begin()));
}
Expr *getUpdateExpr() {
return cast_or_null<Expr>(*std::next(child_begin(), 2));
}
const Expr *getUpdateExpr() const {
return cast_or_null<Expr>(*std::next(child_begin(), 2));
}
bool isXLHSInRHSPart() const { return IsXLHSInRHSPart; }
bool isPostfixUpdate() const { return IsPostfixUpdate; }
Expr *getV() { return cast_or_null<Expr>(*std::next(child_begin(), 3)); }
const Expr *getV() const {
return cast_or_null<Expr>(*std::next(child_begin(), 3));
}
Expr *getExpr() { return cast_or_null<Expr>(*std::next(child_begin(), 4)); }
const Expr *getExpr() const {
return cast_or_null<Expr>(*std::next(child_begin(), 4));
}

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPAtomicDirectiveClass;
}
};

class OMPTargetDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
OMPTargetDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned NumClauses)
: OMPExecutableDirective(this, OMPTargetDirectiveClass, OMPD_target,
StartLoc, EndLoc, NumClauses, 1) {}

explicit OMPTargetDirective(unsigned NumClauses)
: OMPExecutableDirective(this, OMPTargetDirectiveClass, OMPD_target,
SourceLocation(), SourceLocation(), NumClauses,
1) {}

public:
static OMPTargetDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

static OMPTargetDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses, EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTargetDirectiveClass;
}
};

class OMPTargetDataDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
OMPTargetDataDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned NumClauses)
: OMPExecutableDirective(this, OMPTargetDataDirectiveClass,
OMPD_target_data, StartLoc, EndLoc, NumClauses,
1) {}

explicit OMPTargetDataDirective(unsigned NumClauses)
: OMPExecutableDirective(this, OMPTargetDataDirectiveClass,
OMPD_target_data, SourceLocation(),
SourceLocation(), NumClauses, 1) {}

public:
static OMPTargetDataDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

static OMPTargetDataDirective *CreateEmpty(const ASTContext &C, unsigned N,
EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTargetDataDirectiveClass;
}
};

class OMPTargetEnterDataDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
OMPTargetEnterDataDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned NumClauses)
: OMPExecutableDirective(this, OMPTargetEnterDataDirectiveClass,
OMPD_target_enter_data, StartLoc, EndLoc,
NumClauses, 1) {}

explicit OMPTargetEnterDataDirective(unsigned NumClauses)
: OMPExecutableDirective(this, OMPTargetEnterDataDirectiveClass,
OMPD_target_enter_data, SourceLocation(),
SourceLocation(), NumClauses,
1) {}

public:
static OMPTargetEnterDataDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

static OMPTargetEnterDataDirective *CreateEmpty(const ASTContext &C,
unsigned N, EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTargetEnterDataDirectiveClass;
}
};

class OMPTargetExitDataDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
OMPTargetExitDataDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned NumClauses)
: OMPExecutableDirective(this, OMPTargetExitDataDirectiveClass,
OMPD_target_exit_data, StartLoc, EndLoc,
NumClauses, 1) {}

explicit OMPTargetExitDataDirective(unsigned NumClauses)
: OMPExecutableDirective(this, OMPTargetExitDataDirectiveClass,
OMPD_target_exit_data, SourceLocation(),
SourceLocation(), NumClauses,
1) {}

public:
static OMPTargetExitDataDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

static OMPTargetExitDataDirective *CreateEmpty(const ASTContext &C,
unsigned N, EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTargetExitDataDirectiveClass;
}
};

class OMPTargetParallelDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
OMPTargetParallelDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned NumClauses)
: OMPExecutableDirective(this, OMPTargetParallelDirectiveClass,
OMPD_target_parallel, StartLoc, EndLoc,
NumClauses, 1) {}

explicit OMPTargetParallelDirective(unsigned NumClauses)
: OMPExecutableDirective(this, OMPTargetParallelDirectiveClass,
OMPD_target_parallel, SourceLocation(),
SourceLocation(), NumClauses,
1) {}

public:
static OMPTargetParallelDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

static OMPTargetParallelDirective *
CreateEmpty(const ASTContext &C, unsigned NumClauses, EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTargetParallelDirectiveClass;
}
};

class OMPTargetParallelForDirective : public OMPLoopDirective {
friend class ASTStmtReader;

bool HasCancel;

OMPTargetParallelForDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPTargetParallelForDirectiveClass,
OMPD_target_parallel_for, StartLoc, EndLoc,
CollapsedNum, NumClauses),
HasCancel(false) {}

explicit OMPTargetParallelForDirective(unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(this, OMPTargetParallelForDirectiveClass,
OMPD_target_parallel_for, SourceLocation(),
SourceLocation(), CollapsedNum, NumClauses),
HasCancel(false) {}

void setHasCancel(bool Has) { HasCancel = Has; }

public:
static OMPTargetParallelForDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs, bool HasCancel);

static OMPTargetParallelForDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses,
unsigned CollapsedNum,
EmptyShell);

bool hasCancel() const { return HasCancel; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTargetParallelForDirectiveClass;
}
};

class OMPTeamsDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
OMPTeamsDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned NumClauses)
: OMPExecutableDirective(this, OMPTeamsDirectiveClass, OMPD_teams,
StartLoc, EndLoc, NumClauses, 1) {}

explicit OMPTeamsDirective(unsigned NumClauses)
: OMPExecutableDirective(this, OMPTeamsDirectiveClass, OMPD_teams,
SourceLocation(), SourceLocation(), NumClauses,
1) {}

public:
static OMPTeamsDirective *Create(const ASTContext &C, SourceLocation StartLoc,
SourceLocation EndLoc,
ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt);

static OMPTeamsDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses, EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTeamsDirectiveClass;
}
};

class OMPCancellationPointDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
OpenMPDirectiveKind CancelRegion;
OMPCancellationPointDirective(SourceLocation StartLoc, SourceLocation EndLoc)
: OMPExecutableDirective(this, OMPCancellationPointDirectiveClass,
OMPD_cancellation_point, StartLoc, EndLoc, 0, 0),
CancelRegion(OMPD_unknown) {}

explicit OMPCancellationPointDirective()
: OMPExecutableDirective(this, OMPCancellationPointDirectiveClass,
OMPD_cancellation_point, SourceLocation(),
SourceLocation(), 0, 0),
CancelRegion(OMPD_unknown) {}

void setCancelRegion(OpenMPDirectiveKind CR) { CancelRegion = CR; }

public:
static OMPCancellationPointDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
OpenMPDirectiveKind CancelRegion);

static OMPCancellationPointDirective *CreateEmpty(const ASTContext &C,
EmptyShell);

OpenMPDirectiveKind getCancelRegion() const { return CancelRegion; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPCancellationPointDirectiveClass;
}
};

class OMPCancelDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
OpenMPDirectiveKind CancelRegion;
OMPCancelDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned NumClauses)
: OMPExecutableDirective(this, OMPCancelDirectiveClass, OMPD_cancel,
StartLoc, EndLoc, NumClauses, 0),
CancelRegion(OMPD_unknown) {}

explicit OMPCancelDirective(unsigned NumClauses)
: OMPExecutableDirective(this, OMPCancelDirectiveClass, OMPD_cancel,
SourceLocation(), SourceLocation(), NumClauses,
0),
CancelRegion(OMPD_unknown) {}

void setCancelRegion(OpenMPDirectiveKind CR) { CancelRegion = CR; }

public:
static OMPCancelDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
ArrayRef<OMPClause *> Clauses, OpenMPDirectiveKind CancelRegion);

static OMPCancelDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses, EmptyShell);

OpenMPDirectiveKind getCancelRegion() const { return CancelRegion; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPCancelDirectiveClass;
}
};

class OMPTaskLoopDirective : public OMPLoopDirective {
friend class ASTStmtReader;
OMPTaskLoopDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPTaskLoopDirectiveClass, OMPD_taskloop,
StartLoc, EndLoc, CollapsedNum, NumClauses) {}

explicit OMPTaskLoopDirective(unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPTaskLoopDirectiveClass, OMPD_taskloop,
SourceLocation(), SourceLocation(), CollapsedNum,
NumClauses) {}

public:
static OMPTaskLoopDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs);

static OMPTaskLoopDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses,
unsigned CollapsedNum, EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTaskLoopDirectiveClass;
}
};

class OMPTaskLoopSimdDirective : public OMPLoopDirective {
friend class ASTStmtReader;
OMPTaskLoopSimdDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPTaskLoopSimdDirectiveClass,
OMPD_taskloop_simd, StartLoc, EndLoc, CollapsedNum,
NumClauses) {}

explicit OMPTaskLoopSimdDirective(unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPTaskLoopSimdDirectiveClass,
OMPD_taskloop_simd, SourceLocation(), SourceLocation(),
CollapsedNum, NumClauses) {}

public:
static OMPTaskLoopSimdDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs);

static OMPTaskLoopSimdDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses,
unsigned CollapsedNum,
EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTaskLoopSimdDirectiveClass;
}
};

class OMPDistributeDirective : public OMPLoopDirective {
friend class ASTStmtReader;

OMPDistributeDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPDistributeDirectiveClass, OMPD_distribute,
StartLoc, EndLoc, CollapsedNum, NumClauses)
{}

explicit OMPDistributeDirective(unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPDistributeDirectiveClass, OMPD_distribute,
SourceLocation(), SourceLocation(), CollapsedNum,
NumClauses)
{}

public:
static OMPDistributeDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs);

static OMPDistributeDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses,
unsigned CollapsedNum, EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPDistributeDirectiveClass;
}
};

class OMPTargetUpdateDirective : public OMPExecutableDirective {
friend class ASTStmtReader;
OMPTargetUpdateDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned NumClauses)
: OMPExecutableDirective(this, OMPTargetUpdateDirectiveClass,
OMPD_target_update, StartLoc, EndLoc, NumClauses,
1) {}

explicit OMPTargetUpdateDirective(unsigned NumClauses)
: OMPExecutableDirective(this, OMPTargetUpdateDirectiveClass,
OMPD_target_update, SourceLocation(),
SourceLocation(), NumClauses, 1) {}

public:
static OMPTargetUpdateDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt);

static OMPTargetUpdateDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses, EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTargetUpdateDirectiveClass;
}
};

class OMPDistributeParallelForDirective : public OMPLoopDirective {
friend class ASTStmtReader;
bool HasCancel = false;

OMPDistributeParallelForDirective(SourceLocation StartLoc,
SourceLocation EndLoc,
unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPDistributeParallelForDirectiveClass,
OMPD_distribute_parallel_for, StartLoc, EndLoc,
CollapsedNum, NumClauses), HasCancel(false) {}

explicit OMPDistributeParallelForDirective(unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(this, OMPDistributeParallelForDirectiveClass,
OMPD_distribute_parallel_for, SourceLocation(),
SourceLocation(), CollapsedNum, NumClauses),
HasCancel(false) {}

void setHasCancel(bool Has) { HasCancel = Has; }

public:
static OMPDistributeParallelForDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs, bool HasCancel);

static OMPDistributeParallelForDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses,
unsigned CollapsedNum,
EmptyShell);

bool hasCancel() const { return HasCancel; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPDistributeParallelForDirectiveClass;
}
};

class OMPDistributeParallelForSimdDirective final : public OMPLoopDirective {
friend class ASTStmtReader;

OMPDistributeParallelForSimdDirective(SourceLocation StartLoc,
SourceLocation EndLoc,
unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(this, OMPDistributeParallelForSimdDirectiveClass,
OMPD_distribute_parallel_for_simd, StartLoc,
EndLoc, CollapsedNum, NumClauses) {}

explicit OMPDistributeParallelForSimdDirective(unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(this, OMPDistributeParallelForSimdDirectiveClass,
OMPD_distribute_parallel_for_simd,
SourceLocation(), SourceLocation(), CollapsedNum,
NumClauses) {}

public:
static OMPDistributeParallelForSimdDirective *Create(
const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs);

static OMPDistributeParallelForSimdDirective *CreateEmpty(
const ASTContext &C, unsigned NumClauses, unsigned CollapsedNum,
EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPDistributeParallelForSimdDirectiveClass;
}
};

class OMPDistributeSimdDirective final : public OMPLoopDirective {
friend class ASTStmtReader;

OMPDistributeSimdDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPDistributeSimdDirectiveClass,
OMPD_distribute_simd, StartLoc, EndLoc, CollapsedNum,
NumClauses) {}

explicit OMPDistributeSimdDirective(unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(this, OMPDistributeSimdDirectiveClass,
OMPD_distribute_simd, SourceLocation(),
SourceLocation(), CollapsedNum, NumClauses) {}

public:
static OMPDistributeSimdDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs);

static OMPDistributeSimdDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses,
unsigned CollapsedNum,
EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPDistributeSimdDirectiveClass;
}
};

class OMPTargetParallelForSimdDirective final : public OMPLoopDirective {
friend class ASTStmtReader;

OMPTargetParallelForSimdDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPTargetParallelForSimdDirectiveClass,
OMPD_target_parallel_for_simd, StartLoc, EndLoc,
CollapsedNum, NumClauses) {}

explicit OMPTargetParallelForSimdDirective(unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(this, OMPTargetParallelForSimdDirectiveClass,
OMPD_target_parallel_for_simd, SourceLocation(),
SourceLocation(), CollapsedNum, NumClauses) {}

public:
static OMPTargetParallelForSimdDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs);

static OMPTargetParallelForSimdDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses,
unsigned CollapsedNum,
EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTargetParallelForSimdDirectiveClass;
}
};

class OMPTargetSimdDirective final : public OMPLoopDirective {
friend class ASTStmtReader;

OMPTargetSimdDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPTargetSimdDirectiveClass,
OMPD_target_simd, StartLoc, EndLoc, CollapsedNum,
NumClauses) {}

explicit OMPTargetSimdDirective(unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPTargetSimdDirectiveClass, OMPD_target_simd,
SourceLocation(),SourceLocation(), CollapsedNum,
NumClauses) {}

public:
static OMPTargetSimdDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs);

static OMPTargetSimdDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses,
unsigned CollapsedNum,
EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTargetSimdDirectiveClass;
}
};

class OMPTeamsDistributeDirective final : public OMPLoopDirective {
friend class ASTStmtReader;

OMPTeamsDistributeDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPTeamsDistributeDirectiveClass,
OMPD_teams_distribute, StartLoc, EndLoc,
CollapsedNum, NumClauses) {}

explicit OMPTeamsDistributeDirective(unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(this, OMPTeamsDistributeDirectiveClass,
OMPD_teams_distribute, SourceLocation(),
SourceLocation(), CollapsedNum, NumClauses) {}

public:
static OMPTeamsDistributeDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs);

static OMPTeamsDistributeDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses,
unsigned CollapsedNum,
EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTeamsDistributeDirectiveClass;
}
};

class OMPTeamsDistributeSimdDirective final : public OMPLoopDirective {
friend class ASTStmtReader;

OMPTeamsDistributeSimdDirective(SourceLocation StartLoc,
SourceLocation EndLoc, unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(this, OMPTeamsDistributeSimdDirectiveClass,
OMPD_teams_distribute_simd, StartLoc, EndLoc,
CollapsedNum, NumClauses) {}

explicit OMPTeamsDistributeSimdDirective(unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(this, OMPTeamsDistributeSimdDirectiveClass,
OMPD_teams_distribute_simd, SourceLocation(),
SourceLocation(), CollapsedNum, NumClauses) {}

public:
static OMPTeamsDistributeSimdDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs);

static OMPTeamsDistributeSimdDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses,
unsigned CollapsedNum,
EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTeamsDistributeSimdDirectiveClass;
}
};

class OMPTeamsDistributeParallelForSimdDirective final
: public OMPLoopDirective {
friend class ASTStmtReader;

OMPTeamsDistributeParallelForSimdDirective(SourceLocation StartLoc,
SourceLocation EndLoc,
unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(this, OMPTeamsDistributeParallelForSimdDirectiveClass,
OMPD_teams_distribute_parallel_for_simd, StartLoc,
EndLoc, CollapsedNum, NumClauses) {}

explicit OMPTeamsDistributeParallelForSimdDirective(unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(this, OMPTeamsDistributeParallelForSimdDirectiveClass,
OMPD_teams_distribute_parallel_for_simd,
SourceLocation(), SourceLocation(), CollapsedNum,
NumClauses) {}

public:
static OMPTeamsDistributeParallelForSimdDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs);

static OMPTeamsDistributeParallelForSimdDirective *
CreateEmpty(const ASTContext &C, unsigned NumClauses, unsigned CollapsedNum,
EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTeamsDistributeParallelForSimdDirectiveClass;
}
};

class OMPTeamsDistributeParallelForDirective final : public OMPLoopDirective {
friend class ASTStmtReader;
bool HasCancel = false;

OMPTeamsDistributeParallelForDirective(SourceLocation StartLoc,
SourceLocation EndLoc,
unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(this, OMPTeamsDistributeParallelForDirectiveClass,
OMPD_teams_distribute_parallel_for, StartLoc, EndLoc,
CollapsedNum, NumClauses), HasCancel(false) {}

explicit OMPTeamsDistributeParallelForDirective(unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(this, OMPTeamsDistributeParallelForDirectiveClass,
OMPD_teams_distribute_parallel_for, SourceLocation(),
SourceLocation(), CollapsedNum, NumClauses),
HasCancel(false) {}

void setHasCancel(bool Has) { HasCancel = Has; }

public:
static OMPTeamsDistributeParallelForDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs, bool HasCancel);

static OMPTeamsDistributeParallelForDirective *
CreateEmpty(const ASTContext &C, unsigned NumClauses, unsigned CollapsedNum,
EmptyShell);

bool hasCancel() const { return HasCancel; }

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTeamsDistributeParallelForDirectiveClass;
}
};

class OMPTargetTeamsDirective final : public OMPExecutableDirective {
friend class ASTStmtReader;
OMPTargetTeamsDirective(SourceLocation StartLoc, SourceLocation EndLoc,
unsigned NumClauses)
: OMPExecutableDirective(this, OMPTargetTeamsDirectiveClass,
OMPD_target_teams, StartLoc, EndLoc, NumClauses,
1) {}

explicit OMPTargetTeamsDirective(unsigned NumClauses)
: OMPExecutableDirective(this, OMPTargetTeamsDirectiveClass,
OMPD_target_teams, SourceLocation(),
SourceLocation(), NumClauses, 1) {}

public:
static OMPTargetTeamsDirective *Create(const ASTContext &C,
SourceLocation StartLoc,
SourceLocation EndLoc,
ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt);

static OMPTargetTeamsDirective *CreateEmpty(const ASTContext &C,
unsigned NumClauses, EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTargetTeamsDirectiveClass;
}
};

class OMPTargetTeamsDistributeDirective final : public OMPLoopDirective {
friend class ASTStmtReader;

OMPTargetTeamsDistributeDirective(SourceLocation StartLoc,
SourceLocation EndLoc,
unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(this, OMPTargetTeamsDistributeDirectiveClass,
OMPD_target_teams_distribute, StartLoc, EndLoc,
CollapsedNum, NumClauses) {}

explicit OMPTargetTeamsDistributeDirective(unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(this, OMPTargetTeamsDistributeDirectiveClass,
OMPD_target_teams_distribute, SourceLocation(),
SourceLocation(), CollapsedNum, NumClauses) {}

public:
static OMPTargetTeamsDistributeDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs);

static OMPTargetTeamsDistributeDirective *
CreateEmpty(const ASTContext &C, unsigned NumClauses, unsigned CollapsedNum,
EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTargetTeamsDistributeDirectiveClass;
}
};

class OMPTargetTeamsDistributeParallelForDirective final
: public OMPLoopDirective {
friend class ASTStmtReader;
bool HasCancel = false;

OMPTargetTeamsDistributeParallelForDirective(SourceLocation StartLoc,
SourceLocation EndLoc,
unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(this,
OMPTargetTeamsDistributeParallelForDirectiveClass,
OMPD_target_teams_distribute_parallel_for, StartLoc,
EndLoc, CollapsedNum, NumClauses),
HasCancel(false) {}

explicit OMPTargetTeamsDistributeParallelForDirective(unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(
this, OMPTargetTeamsDistributeParallelForDirectiveClass,
OMPD_target_teams_distribute_parallel_for, SourceLocation(),
SourceLocation(), CollapsedNum, NumClauses),
HasCancel(false) {}

void setHasCancel(bool Has) { HasCancel = Has; }

public:
static OMPTargetTeamsDistributeParallelForDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs, bool HasCancel);

static OMPTargetTeamsDistributeParallelForDirective *
CreateEmpty(const ASTContext &C, unsigned NumClauses, unsigned CollapsedNum,
EmptyShell);

bool hasCancel() const { return HasCancel; }

static bool classof(const Stmt *T) {
return T->getStmtClass() ==
OMPTargetTeamsDistributeParallelForDirectiveClass;
}
};

class OMPTargetTeamsDistributeParallelForSimdDirective final
: public OMPLoopDirective {
friend class ASTStmtReader;

OMPTargetTeamsDistributeParallelForSimdDirective(SourceLocation StartLoc,
SourceLocation EndLoc,
unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(this,
OMPTargetTeamsDistributeParallelForSimdDirectiveClass,
OMPD_target_teams_distribute_parallel_for_simd,
StartLoc, EndLoc, CollapsedNum, NumClauses) {}

explicit OMPTargetTeamsDistributeParallelForSimdDirective(
unsigned CollapsedNum, unsigned NumClauses)
: OMPLoopDirective(
this, OMPTargetTeamsDistributeParallelForSimdDirectiveClass,
OMPD_target_teams_distribute_parallel_for_simd, SourceLocation(),
SourceLocation(), CollapsedNum, NumClauses) {}

public:
static OMPTargetTeamsDistributeParallelForSimdDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs);

static OMPTargetTeamsDistributeParallelForSimdDirective *
CreateEmpty(const ASTContext &C, unsigned NumClauses, unsigned CollapsedNum,
EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() ==
OMPTargetTeamsDistributeParallelForSimdDirectiveClass;
}
};

class OMPTargetTeamsDistributeSimdDirective final : public OMPLoopDirective {
friend class ASTStmtReader;

OMPTargetTeamsDistributeSimdDirective(SourceLocation StartLoc,
SourceLocation EndLoc,
unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(this, OMPTargetTeamsDistributeSimdDirectiveClass,
OMPD_target_teams_distribute_simd, StartLoc, EndLoc,
CollapsedNum, NumClauses) {}

explicit OMPTargetTeamsDistributeSimdDirective(unsigned CollapsedNum,
unsigned NumClauses)
: OMPLoopDirective(this, OMPTargetTeamsDistributeSimdDirectiveClass,
OMPD_target_teams_distribute_simd, SourceLocation(),
SourceLocation(), CollapsedNum, NumClauses) {}

public:
static OMPTargetTeamsDistributeSimdDirective *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
unsigned CollapsedNum, ArrayRef<OMPClause *> Clauses,
Stmt *AssociatedStmt, const HelperExprs &Exprs);

static OMPTargetTeamsDistributeSimdDirective *
CreateEmpty(const ASTContext &C, unsigned NumClauses, unsigned CollapsedNum,
EmptyShell);

static bool classof(const Stmt *T) {
return T->getStmtClass() == OMPTargetTeamsDistributeSimdDirectiveClass;
}
};

} 

#endif
