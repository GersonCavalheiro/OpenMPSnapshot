
#ifndef LLVM_CLANG_AST_OPENMPCLAUSE_H
#define LLVM_CLANG_AST_OPENMPCLAUSE_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtIterator.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/OpenMPKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/TrailingObjects.h"
#include <cassert>
#include <cstddef>
#include <iterator>
#include <utility>

namespace clang {

class ASTContext;


class OMPClause {
SourceLocation StartLoc;

SourceLocation EndLoc;

OpenMPClauseKind Kind;

protected:
OMPClause(OpenMPClauseKind K, SourceLocation StartLoc, SourceLocation EndLoc)
: StartLoc(StartLoc), EndLoc(EndLoc), Kind(K) {}

public:
SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const { return StartLoc; }

SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const { return EndLoc; }

void setLocStart(SourceLocation Loc) { StartLoc = Loc; }

void setLocEnd(SourceLocation Loc) { EndLoc = Loc; }

OpenMPClauseKind getClauseKind() const { return Kind; }

bool isImplicit() const { return StartLoc.isInvalid(); }

using child_iterator = StmtIterator;
using const_child_iterator = ConstStmtIterator;
using child_range = llvm::iterator_range<child_iterator>;
using const_child_range = llvm::iterator_range<const_child_iterator>;

child_range children();
const_child_range children() const {
auto Children = const_cast<OMPClause *>(this)->children();
return const_child_range(Children.begin(), Children.end());
}

static bool classof(const OMPClause *) { return true; }
};

class OMPClauseWithPreInit {
friend class OMPClauseReader;

Stmt *PreInit = nullptr;

OpenMPDirectiveKind CaptureRegion = OMPD_unknown;

protected:
OMPClauseWithPreInit(const OMPClause *This) {
assert(get(This) && "get is not tuned for pre-init.");
}

void setPreInitStmt(Stmt *S, OpenMPDirectiveKind ThisRegion = OMPD_unknown) {
PreInit = S;
CaptureRegion = ThisRegion;
}

public:
const Stmt *getPreInitStmt() const { return PreInit; }

Stmt *getPreInitStmt() { return PreInit; }

OpenMPDirectiveKind getCaptureRegion() const { return CaptureRegion; }

static OMPClauseWithPreInit *get(OMPClause *C);
static const OMPClauseWithPreInit *get(const OMPClause *C);
};

class OMPClauseWithPostUpdate : public OMPClauseWithPreInit {
friend class OMPClauseReader;

Expr *PostUpdate = nullptr;

protected:
OMPClauseWithPostUpdate(const OMPClause *This) : OMPClauseWithPreInit(This) {
assert(get(This) && "get is not tuned for post-update.");
}

void setPostUpdateExpr(Expr *S) { PostUpdate = S; }

public:
const Expr *getPostUpdateExpr() const { return PostUpdate; }

Expr *getPostUpdateExpr() { return PostUpdate; }

static OMPClauseWithPostUpdate *get(OMPClause *C);
static const OMPClauseWithPostUpdate *get(const OMPClause *C);
};

template <class T> class OMPVarListClause : public OMPClause {
friend class OMPClauseReader;

SourceLocation LParenLoc;

unsigned NumVars;

protected:
OMPVarListClause(OpenMPClauseKind K, SourceLocation StartLoc,
SourceLocation LParenLoc, SourceLocation EndLoc, unsigned N)
: OMPClause(K, StartLoc, EndLoc), LParenLoc(LParenLoc), NumVars(N) {}

MutableArrayRef<Expr *> getVarRefs() {
return MutableArrayRef<Expr *>(
static_cast<T *>(this)->template getTrailingObjects<Expr *>(), NumVars);
}

void setVarRefs(ArrayRef<Expr *> VL) {
assert(VL.size() == NumVars &&
"Number of variables is not the same as the preallocated buffer");
std::copy(VL.begin(), VL.end(),
static_cast<T *>(this)->template getTrailingObjects<Expr *>());
}

public:
using varlist_iterator = MutableArrayRef<Expr *>::iterator;
using varlist_const_iterator = ArrayRef<const Expr *>::iterator;
using varlist_range = llvm::iterator_range<varlist_iterator>;
using varlist_const_range = llvm::iterator_range<varlist_const_iterator>;

unsigned varlist_size() const { return NumVars; }
bool varlist_empty() const { return NumVars == 0; }

varlist_range varlists() {
return varlist_range(varlist_begin(), varlist_end());
}
varlist_const_range varlists() const {
return varlist_const_range(varlist_begin(), varlist_end());
}

varlist_iterator varlist_begin() { return getVarRefs().begin(); }
varlist_iterator varlist_end() { return getVarRefs().end(); }
varlist_const_iterator varlist_begin() const { return getVarRefs().begin(); }
varlist_const_iterator varlist_end() const { return getVarRefs().end(); }

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

SourceLocation getLParenLoc() const { return LParenLoc; }

ArrayRef<const Expr *> getVarRefs() const {
return llvm::makeArrayRef(
static_cast<const T *>(this)->template getTrailingObjects<Expr *>(),
NumVars);
}
};

class OMPIfClause : public OMPClause, public OMPClauseWithPreInit {
friend class OMPClauseReader;

SourceLocation LParenLoc;

Stmt *Condition = nullptr;

SourceLocation ColonLoc;

OpenMPDirectiveKind NameModifier = OMPD_unknown;

SourceLocation NameModifierLoc;

void setCondition(Expr *Cond) { Condition = Cond; }

void setNameModifier(OpenMPDirectiveKind NM) { NameModifier = NM; }

void setNameModifierLoc(SourceLocation Loc) { NameModifierLoc = Loc; }

void setColonLoc(SourceLocation Loc) { ColonLoc = Loc; }

public:
OMPIfClause(OpenMPDirectiveKind NameModifier, Expr *Cond, Stmt *HelperCond,
OpenMPDirectiveKind CaptureRegion, SourceLocation StartLoc,
SourceLocation LParenLoc, SourceLocation NameModifierLoc,
SourceLocation ColonLoc, SourceLocation EndLoc)
: OMPClause(OMPC_if, StartLoc, EndLoc), OMPClauseWithPreInit(this),
LParenLoc(LParenLoc), Condition(Cond), ColonLoc(ColonLoc),
NameModifier(NameModifier), NameModifierLoc(NameModifierLoc) {
setPreInitStmt(HelperCond, CaptureRegion);
}

OMPIfClause()
: OMPClause(OMPC_if, SourceLocation(), SourceLocation()),
OMPClauseWithPreInit(this) {}

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

SourceLocation getLParenLoc() const { return LParenLoc; }

SourceLocation getColonLoc() const { return ColonLoc; }

Expr *getCondition() const { return cast_or_null<Expr>(Condition); }

OpenMPDirectiveKind getNameModifier() const { return NameModifier; }

SourceLocation getNameModifierLoc() const { return NameModifierLoc; }

child_range children() { return child_range(&Condition, &Condition + 1); }

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_if;
}
};

class OMPFinalClause : public OMPClause {
friend class OMPClauseReader;

SourceLocation LParenLoc;

Stmt *Condition = nullptr;

void setCondition(Expr *Cond) { Condition = Cond; }

public:
OMPFinalClause(Expr *Cond, SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc)
: OMPClause(OMPC_final, StartLoc, EndLoc), LParenLoc(LParenLoc),
Condition(Cond) {}

OMPFinalClause()
: OMPClause(OMPC_final, SourceLocation(), SourceLocation()) {}

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

SourceLocation getLParenLoc() const { return LParenLoc; }

Expr *getCondition() const { return cast_or_null<Expr>(Condition); }

child_range children() { return child_range(&Condition, &Condition + 1); }

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_final;
}
};

class OMPNumThreadsClause : public OMPClause, public OMPClauseWithPreInit {
friend class OMPClauseReader;

SourceLocation LParenLoc;

Stmt *NumThreads = nullptr;

void setNumThreads(Expr *NThreads) { NumThreads = NThreads; }

public:
OMPNumThreadsClause(Expr *NumThreads, Stmt *HelperNumThreads,
OpenMPDirectiveKind CaptureRegion,
SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc)
: OMPClause(OMPC_num_threads, StartLoc, EndLoc),
OMPClauseWithPreInit(this), LParenLoc(LParenLoc),
NumThreads(NumThreads) {
setPreInitStmt(HelperNumThreads, CaptureRegion);
}

OMPNumThreadsClause()
: OMPClause(OMPC_num_threads, SourceLocation(), SourceLocation()),
OMPClauseWithPreInit(this) {}

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

SourceLocation getLParenLoc() const { return LParenLoc; }

Expr *getNumThreads() const { return cast_or_null<Expr>(NumThreads); }

child_range children() { return child_range(&NumThreads, &NumThreads + 1); }

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_num_threads;
}
};

class OMPSafelenClause : public OMPClause {
friend class OMPClauseReader;

SourceLocation LParenLoc;

Stmt *Safelen = nullptr;

void setSafelen(Expr *Len) { Safelen = Len; }

public:
OMPSafelenClause(Expr *Len, SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc)
: OMPClause(OMPC_safelen, StartLoc, EndLoc), LParenLoc(LParenLoc),
Safelen(Len) {}

explicit OMPSafelenClause()
: OMPClause(OMPC_safelen, SourceLocation(), SourceLocation()) {}

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

SourceLocation getLParenLoc() const { return LParenLoc; }

Expr *getSafelen() const { return cast_or_null<Expr>(Safelen); }

child_range children() { return child_range(&Safelen, &Safelen + 1); }

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_safelen;
}
};

class OMPSimdlenClause : public OMPClause {
friend class OMPClauseReader;

SourceLocation LParenLoc;

Stmt *Simdlen = nullptr;

void setSimdlen(Expr *Len) { Simdlen = Len; }

public:
OMPSimdlenClause(Expr *Len, SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc)
: OMPClause(OMPC_simdlen, StartLoc, EndLoc), LParenLoc(LParenLoc),
Simdlen(Len) {}

explicit OMPSimdlenClause()
: OMPClause(OMPC_simdlen, SourceLocation(), SourceLocation()) {}

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

SourceLocation getLParenLoc() const { return LParenLoc; }

Expr *getSimdlen() const { return cast_or_null<Expr>(Simdlen); }

child_range children() { return child_range(&Simdlen, &Simdlen + 1); }

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_simdlen;
}
};

class OMPCollapseClause : public OMPClause {
friend class OMPClauseReader;

SourceLocation LParenLoc;

Stmt *NumForLoops = nullptr;

void setNumForLoops(Expr *Num) { NumForLoops = Num; }

public:
OMPCollapseClause(Expr *Num, SourceLocation StartLoc,
SourceLocation LParenLoc, SourceLocation EndLoc)
: OMPClause(OMPC_collapse, StartLoc, EndLoc), LParenLoc(LParenLoc),
NumForLoops(Num) {}

explicit OMPCollapseClause()
: OMPClause(OMPC_collapse, SourceLocation(), SourceLocation()) {}

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

SourceLocation getLParenLoc() const { return LParenLoc; }

Expr *getNumForLoops() const { return cast_or_null<Expr>(NumForLoops); }

child_range children() { return child_range(&NumForLoops, &NumForLoops + 1); }

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_collapse;
}
};

class OMPDefaultClause : public OMPClause {
friend class OMPClauseReader;

SourceLocation LParenLoc;

OpenMPDefaultClauseKind Kind = OMPC_DEFAULT_unknown;

SourceLocation KindKwLoc;

void setDefaultKind(OpenMPDefaultClauseKind K) { Kind = K; }

void setDefaultKindKwLoc(SourceLocation KLoc) { KindKwLoc = KLoc; }

public:
OMPDefaultClause(OpenMPDefaultClauseKind A, SourceLocation ALoc,
SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc)
: OMPClause(OMPC_default, StartLoc, EndLoc), LParenLoc(LParenLoc),
Kind(A), KindKwLoc(ALoc) {}

OMPDefaultClause()
: OMPClause(OMPC_default, SourceLocation(), SourceLocation()) {}

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

SourceLocation getLParenLoc() const { return LParenLoc; }

OpenMPDefaultClauseKind getDefaultKind() const { return Kind; }

SourceLocation getDefaultKindKwLoc() const { return KindKwLoc; }

child_range children() {
return child_range(child_iterator(), child_iterator());
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_default;
}
};

class OMPProcBindClause : public OMPClause {
friend class OMPClauseReader;

SourceLocation LParenLoc;

OpenMPProcBindClauseKind Kind = OMPC_PROC_BIND_unknown;

SourceLocation KindKwLoc;

void setProcBindKind(OpenMPProcBindClauseKind K) { Kind = K; }

void setProcBindKindKwLoc(SourceLocation KLoc) { KindKwLoc = KLoc; }

public:
OMPProcBindClause(OpenMPProcBindClauseKind A, SourceLocation ALoc,
SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc)
: OMPClause(OMPC_proc_bind, StartLoc, EndLoc), LParenLoc(LParenLoc),
Kind(A), KindKwLoc(ALoc) {}

OMPProcBindClause()
: OMPClause(OMPC_proc_bind, SourceLocation(), SourceLocation()) {}

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

SourceLocation getLParenLoc() const { return LParenLoc; }

OpenMPProcBindClauseKind getProcBindKind() const { return Kind; }

SourceLocation getProcBindKindKwLoc() const { return KindKwLoc; }

child_range children() {
return child_range(child_iterator(), child_iterator());
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_proc_bind;
}
};

class OMPScheduleClause : public OMPClause, public OMPClauseWithPreInit {
friend class OMPClauseReader;

SourceLocation LParenLoc;

OpenMPScheduleClauseKind Kind = OMPC_SCHEDULE_unknown;

enum {FIRST, SECOND, NUM_MODIFIERS};
OpenMPScheduleClauseModifier Modifiers[NUM_MODIFIERS];

SourceLocation ModifiersLoc[NUM_MODIFIERS];

SourceLocation KindLoc;

SourceLocation CommaLoc;

Expr *ChunkSize = nullptr;

void setScheduleKind(OpenMPScheduleClauseKind K) { Kind = K; }

void setFirstScheduleModifier(OpenMPScheduleClauseModifier M) {
Modifiers[FIRST] = M;
}

void setSecondScheduleModifier(OpenMPScheduleClauseModifier M) {
Modifiers[SECOND] = M;
}

void setFirstScheduleModifierLoc(SourceLocation Loc) {
ModifiersLoc[FIRST] = Loc;
}

void setSecondScheduleModifierLoc(SourceLocation Loc) {
ModifiersLoc[SECOND] = Loc;
}

void setScheduleModifer(OpenMPScheduleClauseModifier M) {
if (Modifiers[FIRST] == OMPC_SCHEDULE_MODIFIER_unknown)
Modifiers[FIRST] = M;
else {
assert(Modifiers[SECOND] == OMPC_SCHEDULE_MODIFIER_unknown);
Modifiers[SECOND] = M;
}
}

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

void setScheduleKindLoc(SourceLocation KLoc) { KindLoc = KLoc; }

void setCommaLoc(SourceLocation Loc) { CommaLoc = Loc; }

void setChunkSize(Expr *E) { ChunkSize = E; }

public:
OMPScheduleClause(SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation KLoc, SourceLocation CommaLoc,
SourceLocation EndLoc, OpenMPScheduleClauseKind Kind,
Expr *ChunkSize, Stmt *HelperChunkSize,
OpenMPScheduleClauseModifier M1, SourceLocation M1Loc,
OpenMPScheduleClauseModifier M2, SourceLocation M2Loc)
: OMPClause(OMPC_schedule, StartLoc, EndLoc), OMPClauseWithPreInit(this),
LParenLoc(LParenLoc), Kind(Kind), KindLoc(KLoc), CommaLoc(CommaLoc),
ChunkSize(ChunkSize) {
setPreInitStmt(HelperChunkSize);
Modifiers[FIRST] = M1;
Modifiers[SECOND] = M2;
ModifiersLoc[FIRST] = M1Loc;
ModifiersLoc[SECOND] = M2Loc;
}

explicit OMPScheduleClause()
: OMPClause(OMPC_schedule, SourceLocation(), SourceLocation()),
OMPClauseWithPreInit(this) {
Modifiers[FIRST] = OMPC_SCHEDULE_MODIFIER_unknown;
Modifiers[SECOND] = OMPC_SCHEDULE_MODIFIER_unknown;
}

OpenMPScheduleClauseKind getScheduleKind() const { return Kind; }

OpenMPScheduleClauseModifier getFirstScheduleModifier() const {
return Modifiers[FIRST];
}

OpenMPScheduleClauseModifier getSecondScheduleModifier() const {
return Modifiers[SECOND];
}

SourceLocation getLParenLoc() { return LParenLoc; }

SourceLocation getScheduleKindLoc() { return KindLoc; }

SourceLocation getFirstScheduleModifierLoc() const {
return ModifiersLoc[FIRST];
}

SourceLocation getSecondScheduleModifierLoc() const {
return ModifiersLoc[SECOND];
}

SourceLocation getCommaLoc() { return CommaLoc; }

Expr *getChunkSize() { return ChunkSize; }

const Expr *getChunkSize() const { return ChunkSize; }

child_range children() {
return child_range(reinterpret_cast<Stmt **>(&ChunkSize),
reinterpret_cast<Stmt **>(&ChunkSize) + 1);
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_schedule;
}
};

class OMPOrderedClause final
: public OMPClause,
private llvm::TrailingObjects<OMPOrderedClause, Expr *> {
friend class OMPClauseReader;
friend TrailingObjects;

SourceLocation LParenLoc;

Stmt *NumForLoops = nullptr;

unsigned NumberOfLoops = 0;

OMPOrderedClause(Expr *Num, unsigned NumLoops, SourceLocation StartLoc,
SourceLocation LParenLoc, SourceLocation EndLoc)
: OMPClause(OMPC_ordered, StartLoc, EndLoc), LParenLoc(LParenLoc),
NumForLoops(Num), NumberOfLoops(NumLoops) {}

explicit OMPOrderedClause(unsigned NumLoops)
: OMPClause(OMPC_ordered, SourceLocation(), SourceLocation()),
NumberOfLoops(NumLoops) {}

void setNumForLoops(Expr *Num) { NumForLoops = Num; }

public:
static OMPOrderedClause *Create(const ASTContext &C, Expr *Num,
unsigned NumLoops, SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);

static OMPOrderedClause* CreateEmpty(const ASTContext &C, unsigned NumLoops);

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

SourceLocation getLParenLoc() const { return LParenLoc; }

Expr *getNumForLoops() const { return cast_or_null<Expr>(NumForLoops); }

void setLoopNumIterations(unsigned NumLoop, Expr *NumIterations);
ArrayRef<Expr *> getLoopNumIterations() const;

void setLoopCounter(unsigned NumLoop, Expr *Counter);
Expr *getLoopCunter(unsigned NumLoop);
const Expr *getLoopCunter(unsigned NumLoop) const;

child_range children() { return child_range(&NumForLoops, &NumForLoops + 1); }

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_ordered;
}
};

class OMPNowaitClause : public OMPClause {
public:
OMPNowaitClause(SourceLocation StartLoc, SourceLocation EndLoc)
: OMPClause(OMPC_nowait, StartLoc, EndLoc) {}

OMPNowaitClause()
: OMPClause(OMPC_nowait, SourceLocation(), SourceLocation()) {}

child_range children() {
return child_range(child_iterator(), child_iterator());
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_nowait;
}
};

class OMPUntiedClause : public OMPClause {
public:
OMPUntiedClause(SourceLocation StartLoc, SourceLocation EndLoc)
: OMPClause(OMPC_untied, StartLoc, EndLoc) {}

OMPUntiedClause()
: OMPClause(OMPC_untied, SourceLocation(), SourceLocation()) {}

child_range children() {
return child_range(child_iterator(), child_iterator());
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_untied;
}
};

class OMPMergeableClause : public OMPClause {
public:
OMPMergeableClause(SourceLocation StartLoc, SourceLocation EndLoc)
: OMPClause(OMPC_mergeable, StartLoc, EndLoc) {}

OMPMergeableClause()
: OMPClause(OMPC_mergeable, SourceLocation(), SourceLocation()) {}

child_range children() {
return child_range(child_iterator(), child_iterator());
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_mergeable;
}
};

class OMPReadClause : public OMPClause {
public:
OMPReadClause(SourceLocation StartLoc, SourceLocation EndLoc)
: OMPClause(OMPC_read, StartLoc, EndLoc) {}

OMPReadClause() : OMPClause(OMPC_read, SourceLocation(), SourceLocation()) {}

child_range children() {
return child_range(child_iterator(), child_iterator());
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_read;
}
};

class OMPWriteClause : public OMPClause {
public:
OMPWriteClause(SourceLocation StartLoc, SourceLocation EndLoc)
: OMPClause(OMPC_write, StartLoc, EndLoc) {}

OMPWriteClause()
: OMPClause(OMPC_write, SourceLocation(), SourceLocation()) {}

child_range children() {
return child_range(child_iterator(), child_iterator());
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_write;
}
};

class OMPUpdateClause : public OMPClause {
public:
OMPUpdateClause(SourceLocation StartLoc, SourceLocation EndLoc)
: OMPClause(OMPC_update, StartLoc, EndLoc) {}

OMPUpdateClause()
: OMPClause(OMPC_update, SourceLocation(), SourceLocation()) {}

child_range children() {
return child_range(child_iterator(), child_iterator());
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_update;
}
};

class OMPCaptureClause : public OMPClause {
public:
OMPCaptureClause(SourceLocation StartLoc, SourceLocation EndLoc)
: OMPClause(OMPC_capture, StartLoc, EndLoc) {}

OMPCaptureClause()
: OMPClause(OMPC_capture, SourceLocation(), SourceLocation()) {}

child_range children() {
return child_range(child_iterator(), child_iterator());
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_capture;
}
};

class OMPSeqCstClause : public OMPClause {
public:
OMPSeqCstClause(SourceLocation StartLoc, SourceLocation EndLoc)
: OMPClause(OMPC_seq_cst, StartLoc, EndLoc) {}

OMPSeqCstClause()
: OMPClause(OMPC_seq_cst, SourceLocation(), SourceLocation()) {}

child_range children() {
return child_range(child_iterator(), child_iterator());
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_seq_cst;
}
};

class OMPPrivateClause final
: public OMPVarListClause<OMPPrivateClause>,
private llvm::TrailingObjects<OMPPrivateClause, Expr *> {
friend class OMPClauseReader;
friend OMPVarListClause;
friend TrailingObjects;

OMPPrivateClause(SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc, unsigned N)
: OMPVarListClause<OMPPrivateClause>(OMPC_private, StartLoc, LParenLoc,
EndLoc, N) {}

explicit OMPPrivateClause(unsigned N)
: OMPVarListClause<OMPPrivateClause>(OMPC_private, SourceLocation(),
SourceLocation(), SourceLocation(),
N) {}

void setPrivateCopies(ArrayRef<Expr *> VL);

MutableArrayRef<Expr *> getPrivateCopies() {
return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
}
ArrayRef<const Expr *> getPrivateCopies() const {
return llvm::makeArrayRef(varlist_end(), varlist_size());
}

public:
static OMPPrivateClause *Create(const ASTContext &C, SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc, ArrayRef<Expr *> VL,
ArrayRef<Expr *> PrivateVL);

static OMPPrivateClause *CreateEmpty(const ASTContext &C, unsigned N);

using private_copies_iterator = MutableArrayRef<Expr *>::iterator;
using private_copies_const_iterator = ArrayRef<const Expr *>::iterator;
using private_copies_range = llvm::iterator_range<private_copies_iterator>;
using private_copies_const_range =
llvm::iterator_range<private_copies_const_iterator>;

private_copies_range private_copies() {
return private_copies_range(getPrivateCopies().begin(),
getPrivateCopies().end());
}

private_copies_const_range private_copies() const {
return private_copies_const_range(getPrivateCopies().begin(),
getPrivateCopies().end());
}

child_range children() {
return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
reinterpret_cast<Stmt **>(varlist_end()));
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_private;
}
};

class OMPFirstprivateClause final
: public OMPVarListClause<OMPFirstprivateClause>,
public OMPClauseWithPreInit,
private llvm::TrailingObjects<OMPFirstprivateClause, Expr *> {
friend class OMPClauseReader;
friend OMPVarListClause;
friend TrailingObjects;

OMPFirstprivateClause(SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc, unsigned N)
: OMPVarListClause<OMPFirstprivateClause>(OMPC_firstprivate, StartLoc,
LParenLoc, EndLoc, N),
OMPClauseWithPreInit(this) {}

explicit OMPFirstprivateClause(unsigned N)
: OMPVarListClause<OMPFirstprivateClause>(
OMPC_firstprivate, SourceLocation(), SourceLocation(),
SourceLocation(), N),
OMPClauseWithPreInit(this) {}

void setPrivateCopies(ArrayRef<Expr *> VL);

MutableArrayRef<Expr *> getPrivateCopies() {
return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
}
ArrayRef<const Expr *> getPrivateCopies() const {
return llvm::makeArrayRef(varlist_end(), varlist_size());
}

void setInits(ArrayRef<Expr *> VL);

MutableArrayRef<Expr *> getInits() {
return MutableArrayRef<Expr *>(getPrivateCopies().end(), varlist_size());
}
ArrayRef<const Expr *> getInits() const {
return llvm::makeArrayRef(getPrivateCopies().end(), varlist_size());
}

public:
static OMPFirstprivateClause *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc, ArrayRef<Expr *> VL, ArrayRef<Expr *> PrivateVL,
ArrayRef<Expr *> InitVL, Stmt *PreInit);

static OMPFirstprivateClause *CreateEmpty(const ASTContext &C, unsigned N);

using private_copies_iterator = MutableArrayRef<Expr *>::iterator;
using private_copies_const_iterator = ArrayRef<const Expr *>::iterator;
using private_copies_range = llvm::iterator_range<private_copies_iterator>;
using private_copies_const_range =
llvm::iterator_range<private_copies_const_iterator>;

private_copies_range private_copies() {
return private_copies_range(getPrivateCopies().begin(),
getPrivateCopies().end());
}
private_copies_const_range private_copies() const {
return private_copies_const_range(getPrivateCopies().begin(),
getPrivateCopies().end());
}

using inits_iterator = MutableArrayRef<Expr *>::iterator;
using inits_const_iterator = ArrayRef<const Expr *>::iterator;
using inits_range = llvm::iterator_range<inits_iterator>;
using inits_const_range = llvm::iterator_range<inits_const_iterator>;

inits_range inits() {
return inits_range(getInits().begin(), getInits().end());
}
inits_const_range inits() const {
return inits_const_range(getInits().begin(), getInits().end());
}

child_range children() {
return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
reinterpret_cast<Stmt **>(varlist_end()));
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_firstprivate;
}
};

class OMPLastprivateClause final
: public OMPVarListClause<OMPLastprivateClause>,
public OMPClauseWithPostUpdate,
private llvm::TrailingObjects<OMPLastprivateClause, Expr *> {
friend class OMPClauseReader;
friend OMPVarListClause;
friend TrailingObjects;

OMPLastprivateClause(SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc, unsigned N)
: OMPVarListClause<OMPLastprivateClause>(OMPC_lastprivate, StartLoc,
LParenLoc, EndLoc, N),
OMPClauseWithPostUpdate(this) {}

explicit OMPLastprivateClause(unsigned N)
: OMPVarListClause<OMPLastprivateClause>(
OMPC_lastprivate, SourceLocation(), SourceLocation(),
SourceLocation(), N),
OMPClauseWithPostUpdate(this) {}

MutableArrayRef<Expr *> getPrivateCopies() {
return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
}
ArrayRef<const Expr *> getPrivateCopies() const {
return llvm::makeArrayRef(varlist_end(), varlist_size());
}

void setSourceExprs(ArrayRef<Expr *> SrcExprs);

MutableArrayRef<Expr *> getSourceExprs() {
return MutableArrayRef<Expr *>(getPrivateCopies().end(), varlist_size());
}
ArrayRef<const Expr *> getSourceExprs() const {
return llvm::makeArrayRef(getPrivateCopies().end(), varlist_size());
}

void setDestinationExprs(ArrayRef<Expr *> DstExprs);

MutableArrayRef<Expr *> getDestinationExprs() {
return MutableArrayRef<Expr *>(getSourceExprs().end(), varlist_size());
}
ArrayRef<const Expr *> getDestinationExprs() const {
return llvm::makeArrayRef(getSourceExprs().end(), varlist_size());
}

void setAssignmentOps(ArrayRef<Expr *> AssignmentOps);

MutableArrayRef<Expr *> getAssignmentOps() {
return MutableArrayRef<Expr *>(getDestinationExprs().end(), varlist_size());
}
ArrayRef<const Expr *> getAssignmentOps() const {
return llvm::makeArrayRef(getDestinationExprs().end(), varlist_size());
}

public:
static OMPLastprivateClause *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc, ArrayRef<Expr *> VL, ArrayRef<Expr *> SrcExprs,
ArrayRef<Expr *> DstExprs, ArrayRef<Expr *> AssignmentOps,
Stmt *PreInit, Expr *PostUpdate);

static OMPLastprivateClause *CreateEmpty(const ASTContext &C, unsigned N);

using helper_expr_iterator = MutableArrayRef<Expr *>::iterator;
using helper_expr_const_iterator = ArrayRef<const Expr *>::iterator;
using helper_expr_range = llvm::iterator_range<helper_expr_iterator>;
using helper_expr_const_range =
llvm::iterator_range<helper_expr_const_iterator>;

void setPrivateCopies(ArrayRef<Expr *> PrivateCopies);

helper_expr_const_range private_copies() const {
return helper_expr_const_range(getPrivateCopies().begin(),
getPrivateCopies().end());
}

helper_expr_range private_copies() {
return helper_expr_range(getPrivateCopies().begin(),
getPrivateCopies().end());
}

helper_expr_const_range source_exprs() const {
return helper_expr_const_range(getSourceExprs().begin(),
getSourceExprs().end());
}

helper_expr_range source_exprs() {
return helper_expr_range(getSourceExprs().begin(), getSourceExprs().end());
}

helper_expr_const_range destination_exprs() const {
return helper_expr_const_range(getDestinationExprs().begin(),
getDestinationExprs().end());
}

helper_expr_range destination_exprs() {
return helper_expr_range(getDestinationExprs().begin(),
getDestinationExprs().end());
}

helper_expr_const_range assignment_ops() const {
return helper_expr_const_range(getAssignmentOps().begin(),
getAssignmentOps().end());
}

helper_expr_range assignment_ops() {
return helper_expr_range(getAssignmentOps().begin(),
getAssignmentOps().end());
}

child_range children() {
return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
reinterpret_cast<Stmt **>(varlist_end()));
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_lastprivate;
}
};

class OMPSharedClause final
: public OMPVarListClause<OMPSharedClause>,
private llvm::TrailingObjects<OMPSharedClause, Expr *> {
friend OMPVarListClause;
friend TrailingObjects;

OMPSharedClause(SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc, unsigned N)
: OMPVarListClause<OMPSharedClause>(OMPC_shared, StartLoc, LParenLoc,
EndLoc, N) {}

explicit OMPSharedClause(unsigned N)
: OMPVarListClause<OMPSharedClause>(OMPC_shared, SourceLocation(),
SourceLocation(), SourceLocation(),
N) {}

public:
static OMPSharedClause *Create(const ASTContext &C, SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc, ArrayRef<Expr *> VL);

static OMPSharedClause *CreateEmpty(const ASTContext &C, unsigned N);

child_range children() {
return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
reinterpret_cast<Stmt **>(varlist_end()));
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_shared;
}
};

class OMPReductionClause final
: public OMPVarListClause<OMPReductionClause>,
public OMPClauseWithPostUpdate,
private llvm::TrailingObjects<OMPReductionClause, Expr *> {
friend class OMPClauseReader;
friend OMPVarListClause;
friend TrailingObjects;

SourceLocation ColonLoc;

NestedNameSpecifierLoc QualifierLoc;

DeclarationNameInfo NameInfo;

OMPReductionClause(SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation ColonLoc, SourceLocation EndLoc, unsigned N,
NestedNameSpecifierLoc QualifierLoc,
const DeclarationNameInfo &NameInfo)
: OMPVarListClause<OMPReductionClause>(OMPC_reduction, StartLoc,
LParenLoc, EndLoc, N),
OMPClauseWithPostUpdate(this), ColonLoc(ColonLoc),
QualifierLoc(QualifierLoc), NameInfo(NameInfo) {}

explicit OMPReductionClause(unsigned N)
: OMPVarListClause<OMPReductionClause>(OMPC_reduction, SourceLocation(),
SourceLocation(), SourceLocation(),
N),
OMPClauseWithPostUpdate(this) {}

void setColonLoc(SourceLocation CL) { ColonLoc = CL; }

void setNameInfo(DeclarationNameInfo DNI) { NameInfo = DNI; }

void setQualifierLoc(NestedNameSpecifierLoc NSL) { QualifierLoc = NSL; }

void setPrivates(ArrayRef<Expr *> Privates);

MutableArrayRef<Expr *> getPrivates() {
return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
}
ArrayRef<const Expr *> getPrivates() const {
return llvm::makeArrayRef(varlist_end(), varlist_size());
}

void setLHSExprs(ArrayRef<Expr *> LHSExprs);

MutableArrayRef<Expr *> getLHSExprs() {
return MutableArrayRef<Expr *>(getPrivates().end(), varlist_size());
}
ArrayRef<const Expr *> getLHSExprs() const {
return llvm::makeArrayRef(getPrivates().end(), varlist_size());
}

void setRHSExprs(ArrayRef<Expr *> RHSExprs);

MutableArrayRef<Expr *> getRHSExprs() {
return MutableArrayRef<Expr *>(getLHSExprs().end(), varlist_size());
}
ArrayRef<const Expr *> getRHSExprs() const {
return llvm::makeArrayRef(getLHSExprs().end(), varlist_size());
}

void setReductionOps(ArrayRef<Expr *> ReductionOps);

MutableArrayRef<Expr *> getReductionOps() {
return MutableArrayRef<Expr *>(getRHSExprs().end(), varlist_size());
}
ArrayRef<const Expr *> getReductionOps() const {
return llvm::makeArrayRef(getRHSExprs().end(), varlist_size());
}

public:
static OMPReductionClause *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation ColonLoc, SourceLocation EndLoc, ArrayRef<Expr *> VL,
NestedNameSpecifierLoc QualifierLoc,
const DeclarationNameInfo &NameInfo, ArrayRef<Expr *> Privates,
ArrayRef<Expr *> LHSExprs, ArrayRef<Expr *> RHSExprs,
ArrayRef<Expr *> ReductionOps, Stmt *PreInit, Expr *PostUpdate);

static OMPReductionClause *CreateEmpty(const ASTContext &C, unsigned N);

SourceLocation getColonLoc() const { return ColonLoc; }

const DeclarationNameInfo &getNameInfo() const { return NameInfo; }

NestedNameSpecifierLoc getQualifierLoc() const { return QualifierLoc; }

using helper_expr_iterator = MutableArrayRef<Expr *>::iterator;
using helper_expr_const_iterator = ArrayRef<const Expr *>::iterator;
using helper_expr_range = llvm::iterator_range<helper_expr_iterator>;
using helper_expr_const_range =
llvm::iterator_range<helper_expr_const_iterator>;

helper_expr_const_range privates() const {
return helper_expr_const_range(getPrivates().begin(), getPrivates().end());
}

helper_expr_range privates() {
return helper_expr_range(getPrivates().begin(), getPrivates().end());
}

helper_expr_const_range lhs_exprs() const {
return helper_expr_const_range(getLHSExprs().begin(), getLHSExprs().end());
}

helper_expr_range lhs_exprs() {
return helper_expr_range(getLHSExprs().begin(), getLHSExprs().end());
}

helper_expr_const_range rhs_exprs() const {
return helper_expr_const_range(getRHSExprs().begin(), getRHSExprs().end());
}

helper_expr_range rhs_exprs() {
return helper_expr_range(getRHSExprs().begin(), getRHSExprs().end());
}

helper_expr_const_range reduction_ops() const {
return helper_expr_const_range(getReductionOps().begin(),
getReductionOps().end());
}

helper_expr_range reduction_ops() {
return helper_expr_range(getReductionOps().begin(),
getReductionOps().end());
}

child_range children() {
return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
reinterpret_cast<Stmt **>(varlist_end()));
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_reduction;
}
};

class OMPTaskReductionClause final
: public OMPVarListClause<OMPTaskReductionClause>,
public OMPClauseWithPostUpdate,
private llvm::TrailingObjects<OMPTaskReductionClause, Expr *> {
friend class OMPClauseReader;
friend OMPVarListClause;
friend TrailingObjects;

SourceLocation ColonLoc;

NestedNameSpecifierLoc QualifierLoc;

DeclarationNameInfo NameInfo;

OMPTaskReductionClause(SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation ColonLoc, SourceLocation EndLoc,
unsigned N, NestedNameSpecifierLoc QualifierLoc,
const DeclarationNameInfo &NameInfo)
: OMPVarListClause<OMPTaskReductionClause>(OMPC_task_reduction, StartLoc,
LParenLoc, EndLoc, N),
OMPClauseWithPostUpdate(this), ColonLoc(ColonLoc),
QualifierLoc(QualifierLoc), NameInfo(NameInfo) {}

explicit OMPTaskReductionClause(unsigned N)
: OMPVarListClause<OMPTaskReductionClause>(
OMPC_task_reduction, SourceLocation(), SourceLocation(),
SourceLocation(), N),
OMPClauseWithPostUpdate(this) {}

void setColonLoc(SourceLocation CL) { ColonLoc = CL; }

void setNameInfo(DeclarationNameInfo DNI) { NameInfo = DNI; }

void setQualifierLoc(NestedNameSpecifierLoc NSL) { QualifierLoc = NSL; }

void setPrivates(ArrayRef<Expr *> Privates);

MutableArrayRef<Expr *> getPrivates() {
return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
}
ArrayRef<const Expr *> getPrivates() const {
return llvm::makeArrayRef(varlist_end(), varlist_size());
}

void setLHSExprs(ArrayRef<Expr *> LHSExprs);

MutableArrayRef<Expr *> getLHSExprs() {
return MutableArrayRef<Expr *>(getPrivates().end(), varlist_size());
}
ArrayRef<const Expr *> getLHSExprs() const {
return llvm::makeArrayRef(getPrivates().end(), varlist_size());
}

void setRHSExprs(ArrayRef<Expr *> RHSExprs);

MutableArrayRef<Expr *> getRHSExprs() {
return MutableArrayRef<Expr *>(getLHSExprs().end(), varlist_size());
}
ArrayRef<const Expr *> getRHSExprs() const {
return llvm::makeArrayRef(getLHSExprs().end(), varlist_size());
}

void setReductionOps(ArrayRef<Expr *> ReductionOps);

MutableArrayRef<Expr *> getReductionOps() {
return MutableArrayRef<Expr *>(getRHSExprs().end(), varlist_size());
}
ArrayRef<const Expr *> getReductionOps() const {
return llvm::makeArrayRef(getRHSExprs().end(), varlist_size());
}

public:
static OMPTaskReductionClause *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation ColonLoc, SourceLocation EndLoc, ArrayRef<Expr *> VL,
NestedNameSpecifierLoc QualifierLoc,
const DeclarationNameInfo &NameInfo, ArrayRef<Expr *> Privates,
ArrayRef<Expr *> LHSExprs, ArrayRef<Expr *> RHSExprs,
ArrayRef<Expr *> ReductionOps, Stmt *PreInit, Expr *PostUpdate);

static OMPTaskReductionClause *CreateEmpty(const ASTContext &C, unsigned N);

SourceLocation getColonLoc() const { return ColonLoc; }

const DeclarationNameInfo &getNameInfo() const { return NameInfo; }

NestedNameSpecifierLoc getQualifierLoc() const { return QualifierLoc; }

using helper_expr_iterator = MutableArrayRef<Expr *>::iterator;
using helper_expr_const_iterator = ArrayRef<const Expr *>::iterator;
using helper_expr_range = llvm::iterator_range<helper_expr_iterator>;
using helper_expr_const_range =
llvm::iterator_range<helper_expr_const_iterator>;

helper_expr_const_range privates() const {
return helper_expr_const_range(getPrivates().begin(), getPrivates().end());
}

helper_expr_range privates() {
return helper_expr_range(getPrivates().begin(), getPrivates().end());
}

helper_expr_const_range lhs_exprs() const {
return helper_expr_const_range(getLHSExprs().begin(), getLHSExprs().end());
}

helper_expr_range lhs_exprs() {
return helper_expr_range(getLHSExprs().begin(), getLHSExprs().end());
}

helper_expr_const_range rhs_exprs() const {
return helper_expr_const_range(getRHSExprs().begin(), getRHSExprs().end());
}

helper_expr_range rhs_exprs() {
return helper_expr_range(getRHSExprs().begin(), getRHSExprs().end());
}

helper_expr_const_range reduction_ops() const {
return helper_expr_const_range(getReductionOps().begin(),
getReductionOps().end());
}

helper_expr_range reduction_ops() {
return helper_expr_range(getReductionOps().begin(),
getReductionOps().end());
}

child_range children() {
return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
reinterpret_cast<Stmt **>(varlist_end()));
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_task_reduction;
}
};

class OMPInReductionClause final
: public OMPVarListClause<OMPInReductionClause>,
public OMPClauseWithPostUpdate,
private llvm::TrailingObjects<OMPInReductionClause, Expr *> {
friend class OMPClauseReader;
friend OMPVarListClause;
friend TrailingObjects;

SourceLocation ColonLoc;

NestedNameSpecifierLoc QualifierLoc;

DeclarationNameInfo NameInfo;

OMPInReductionClause(SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation ColonLoc, SourceLocation EndLoc,
unsigned N, NestedNameSpecifierLoc QualifierLoc,
const DeclarationNameInfo &NameInfo)
: OMPVarListClause<OMPInReductionClause>(OMPC_in_reduction, StartLoc,
LParenLoc, EndLoc, N),
OMPClauseWithPostUpdate(this), ColonLoc(ColonLoc),
QualifierLoc(QualifierLoc), NameInfo(NameInfo) {}

explicit OMPInReductionClause(unsigned N)
: OMPVarListClause<OMPInReductionClause>(
OMPC_in_reduction, SourceLocation(), SourceLocation(),
SourceLocation(), N),
OMPClauseWithPostUpdate(this) {}

void setColonLoc(SourceLocation CL) { ColonLoc = CL; }

void setNameInfo(DeclarationNameInfo DNI) { NameInfo = DNI; }

void setQualifierLoc(NestedNameSpecifierLoc NSL) { QualifierLoc = NSL; }

void setPrivates(ArrayRef<Expr *> Privates);

MutableArrayRef<Expr *> getPrivates() {
return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
}
ArrayRef<const Expr *> getPrivates() const {
return llvm::makeArrayRef(varlist_end(), varlist_size());
}

void setLHSExprs(ArrayRef<Expr *> LHSExprs);

MutableArrayRef<Expr *> getLHSExprs() {
return MutableArrayRef<Expr *>(getPrivates().end(), varlist_size());
}
ArrayRef<const Expr *> getLHSExprs() const {
return llvm::makeArrayRef(getPrivates().end(), varlist_size());
}

void setRHSExprs(ArrayRef<Expr *> RHSExprs);

MutableArrayRef<Expr *> getRHSExprs() {
return MutableArrayRef<Expr *>(getLHSExprs().end(), varlist_size());
}
ArrayRef<const Expr *> getRHSExprs() const {
return llvm::makeArrayRef(getLHSExprs().end(), varlist_size());
}

void setReductionOps(ArrayRef<Expr *> ReductionOps);

MutableArrayRef<Expr *> getReductionOps() {
return MutableArrayRef<Expr *>(getRHSExprs().end(), varlist_size());
}
ArrayRef<const Expr *> getReductionOps() const {
return llvm::makeArrayRef(getRHSExprs().end(), varlist_size());
}

void setTaskgroupDescriptors(ArrayRef<Expr *> ReductionOps);

MutableArrayRef<Expr *> getTaskgroupDescriptors() {
return MutableArrayRef<Expr *>(getReductionOps().end(), varlist_size());
}
ArrayRef<const Expr *> getTaskgroupDescriptors() const {
return llvm::makeArrayRef(getReductionOps().end(), varlist_size());
}

public:
static OMPInReductionClause *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation ColonLoc, SourceLocation EndLoc, ArrayRef<Expr *> VL,
NestedNameSpecifierLoc QualifierLoc,
const DeclarationNameInfo &NameInfo, ArrayRef<Expr *> Privates,
ArrayRef<Expr *> LHSExprs, ArrayRef<Expr *> RHSExprs,
ArrayRef<Expr *> ReductionOps, ArrayRef<Expr *> TaskgroupDescriptors,
Stmt *PreInit, Expr *PostUpdate);

static OMPInReductionClause *CreateEmpty(const ASTContext &C, unsigned N);

SourceLocation getColonLoc() const { return ColonLoc; }

const DeclarationNameInfo &getNameInfo() const { return NameInfo; }

NestedNameSpecifierLoc getQualifierLoc() const { return QualifierLoc; }

using helper_expr_iterator = MutableArrayRef<Expr *>::iterator;
using helper_expr_const_iterator = ArrayRef<const Expr *>::iterator;
using helper_expr_range = llvm::iterator_range<helper_expr_iterator>;
using helper_expr_const_range =
llvm::iterator_range<helper_expr_const_iterator>;

helper_expr_const_range privates() const {
return helper_expr_const_range(getPrivates().begin(), getPrivates().end());
}

helper_expr_range privates() {
return helper_expr_range(getPrivates().begin(), getPrivates().end());
}

helper_expr_const_range lhs_exprs() const {
return helper_expr_const_range(getLHSExprs().begin(), getLHSExprs().end());
}

helper_expr_range lhs_exprs() {
return helper_expr_range(getLHSExprs().begin(), getLHSExprs().end());
}

helper_expr_const_range rhs_exprs() const {
return helper_expr_const_range(getRHSExprs().begin(), getRHSExprs().end());
}

helper_expr_range rhs_exprs() {
return helper_expr_range(getRHSExprs().begin(), getRHSExprs().end());
}

helper_expr_const_range reduction_ops() const {
return helper_expr_const_range(getReductionOps().begin(),
getReductionOps().end());
}

helper_expr_range reduction_ops() {
return helper_expr_range(getReductionOps().begin(),
getReductionOps().end());
}

helper_expr_const_range taskgroup_descriptors() const {
return helper_expr_const_range(getTaskgroupDescriptors().begin(),
getTaskgroupDescriptors().end());
}

helper_expr_range taskgroup_descriptors() {
return helper_expr_range(getTaskgroupDescriptors().begin(),
getTaskgroupDescriptors().end());
}

child_range children() {
return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
reinterpret_cast<Stmt **>(varlist_end()));
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_in_reduction;
}
};

class OMPLinearClause final
: public OMPVarListClause<OMPLinearClause>,
public OMPClauseWithPostUpdate,
private llvm::TrailingObjects<OMPLinearClause, Expr *> {
friend class OMPClauseReader;
friend OMPVarListClause;
friend TrailingObjects;

OpenMPLinearClauseKind Modifier = OMPC_LINEAR_val;

SourceLocation ModifierLoc;

SourceLocation ColonLoc;

void setStep(Expr *Step) { *(getFinals().end()) = Step; }

void setCalcStep(Expr *CalcStep) { *(getFinals().end() + 1) = CalcStep; }

OMPLinearClause(SourceLocation StartLoc, SourceLocation LParenLoc,
OpenMPLinearClauseKind Modifier, SourceLocation ModifierLoc,
SourceLocation ColonLoc, SourceLocation EndLoc,
unsigned NumVars)
: OMPVarListClause<OMPLinearClause>(OMPC_linear, StartLoc, LParenLoc,
EndLoc, NumVars),
OMPClauseWithPostUpdate(this), Modifier(Modifier),
ModifierLoc(ModifierLoc), ColonLoc(ColonLoc) {}

explicit OMPLinearClause(unsigned NumVars)
: OMPVarListClause<OMPLinearClause>(OMPC_linear, SourceLocation(),
SourceLocation(), SourceLocation(),
NumVars),
OMPClauseWithPostUpdate(this) {}

MutableArrayRef<Expr *> getPrivates() {
return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
}
ArrayRef<const Expr *> getPrivates() const {
return llvm::makeArrayRef(varlist_end(), varlist_size());
}

MutableArrayRef<Expr *> getInits() {
return MutableArrayRef<Expr *>(getPrivates().end(), varlist_size());
}
ArrayRef<const Expr *> getInits() const {
return llvm::makeArrayRef(getPrivates().end(), varlist_size());
}

MutableArrayRef<Expr *> getUpdates() {
return MutableArrayRef<Expr *>(getInits().end(), varlist_size());
}
ArrayRef<const Expr *> getUpdates() const {
return llvm::makeArrayRef(getInits().end(), varlist_size());
}

MutableArrayRef<Expr *> getFinals() {
return MutableArrayRef<Expr *>(getUpdates().end(), varlist_size());
}
ArrayRef<const Expr *> getFinals() const {
return llvm::makeArrayRef(getUpdates().end(), varlist_size());
}

void setPrivates(ArrayRef<Expr *> PL);

void setInits(ArrayRef<Expr *> IL);

public:
static OMPLinearClause *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
OpenMPLinearClauseKind Modifier, SourceLocation ModifierLoc,
SourceLocation ColonLoc, SourceLocation EndLoc, ArrayRef<Expr *> VL,
ArrayRef<Expr *> PL, ArrayRef<Expr *> IL, Expr *Step, Expr *CalcStep,
Stmt *PreInit, Expr *PostUpdate);

static OMPLinearClause *CreateEmpty(const ASTContext &C, unsigned NumVars);

void setModifier(OpenMPLinearClauseKind Kind) { Modifier = Kind; }

OpenMPLinearClauseKind getModifier() const { return Modifier; }

void setModifierLoc(SourceLocation Loc) { ModifierLoc = Loc; }

SourceLocation getModifierLoc() const { return ModifierLoc; }

void setColonLoc(SourceLocation Loc) { ColonLoc = Loc; }

SourceLocation getColonLoc() const { return ColonLoc; }

Expr *getStep() { return *(getFinals().end()); }

const Expr *getStep() const { return *(getFinals().end()); }

Expr *getCalcStep() { return *(getFinals().end() + 1); }

const Expr *getCalcStep() const { return *(getFinals().end() + 1); }

void setUpdates(ArrayRef<Expr *> UL);

void setFinals(ArrayRef<Expr *> FL);

using privates_iterator = MutableArrayRef<Expr *>::iterator;
using privates_const_iterator = ArrayRef<const Expr *>::iterator;
using privates_range = llvm::iterator_range<privates_iterator>;
using privates_const_range = llvm::iterator_range<privates_const_iterator>;

privates_range privates() {
return privates_range(getPrivates().begin(), getPrivates().end());
}

privates_const_range privates() const {
return privates_const_range(getPrivates().begin(), getPrivates().end());
}

using inits_iterator = MutableArrayRef<Expr *>::iterator;
using inits_const_iterator = ArrayRef<const Expr *>::iterator;
using inits_range = llvm::iterator_range<inits_iterator>;
using inits_const_range = llvm::iterator_range<inits_const_iterator>;

inits_range inits() {
return inits_range(getInits().begin(), getInits().end());
}

inits_const_range inits() const {
return inits_const_range(getInits().begin(), getInits().end());
}

using updates_iterator = MutableArrayRef<Expr *>::iterator;
using updates_const_iterator = ArrayRef<const Expr *>::iterator;
using updates_range = llvm::iterator_range<updates_iterator>;
using updates_const_range = llvm::iterator_range<updates_const_iterator>;

updates_range updates() {
return updates_range(getUpdates().begin(), getUpdates().end());
}

updates_const_range updates() const {
return updates_const_range(getUpdates().begin(), getUpdates().end());
}

using finals_iterator = MutableArrayRef<Expr *>::iterator;
using finals_const_iterator = ArrayRef<const Expr *>::iterator;
using finals_range = llvm::iterator_range<finals_iterator>;
using finals_const_range = llvm::iterator_range<finals_const_iterator>;

finals_range finals() {
return finals_range(getFinals().begin(), getFinals().end());
}

finals_const_range finals() const {
return finals_const_range(getFinals().begin(), getFinals().end());
}

child_range children() {
return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
reinterpret_cast<Stmt **>(varlist_end()));
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_linear;
}
};

class OMPAlignedClause final
: public OMPVarListClause<OMPAlignedClause>,
private llvm::TrailingObjects<OMPAlignedClause, Expr *> {
friend class OMPClauseReader;
friend OMPVarListClause;
friend TrailingObjects;

SourceLocation ColonLoc;

void setAlignment(Expr *A) { *varlist_end() = A; }

OMPAlignedClause(SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation ColonLoc, SourceLocation EndLoc,
unsigned NumVars)
: OMPVarListClause<OMPAlignedClause>(OMPC_aligned, StartLoc, LParenLoc,
EndLoc, NumVars),
ColonLoc(ColonLoc) {}

explicit OMPAlignedClause(unsigned NumVars)
: OMPVarListClause<OMPAlignedClause>(OMPC_aligned, SourceLocation(),
SourceLocation(), SourceLocation(),
NumVars) {}

public:
static OMPAlignedClause *Create(const ASTContext &C, SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation ColonLoc,
SourceLocation EndLoc, ArrayRef<Expr *> VL,
Expr *A);

static OMPAlignedClause *CreateEmpty(const ASTContext &C, unsigned NumVars);

void setColonLoc(SourceLocation Loc) { ColonLoc = Loc; }

SourceLocation getColonLoc() const { return ColonLoc; }

Expr *getAlignment() { return *varlist_end(); }

const Expr *getAlignment() const { return *varlist_end(); }

child_range children() {
return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
reinterpret_cast<Stmt **>(varlist_end()));
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_aligned;
}
};

class OMPCopyinClause final
: public OMPVarListClause<OMPCopyinClause>,
private llvm::TrailingObjects<OMPCopyinClause, Expr *> {

friend class OMPClauseReader;
friend OMPVarListClause;
friend TrailingObjects;

OMPCopyinClause(SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc, unsigned N)
: OMPVarListClause<OMPCopyinClause>(OMPC_copyin, StartLoc, LParenLoc,
EndLoc, N) {}

explicit OMPCopyinClause(unsigned N)
: OMPVarListClause<OMPCopyinClause>(OMPC_copyin, SourceLocation(),
SourceLocation(), SourceLocation(),
N) {}

void setSourceExprs(ArrayRef<Expr *> SrcExprs);

MutableArrayRef<Expr *> getSourceExprs() {
return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
}
ArrayRef<const Expr *> getSourceExprs() const {
return llvm::makeArrayRef(varlist_end(), varlist_size());
}

void setDestinationExprs(ArrayRef<Expr *> DstExprs);

MutableArrayRef<Expr *> getDestinationExprs() {
return MutableArrayRef<Expr *>(getSourceExprs().end(), varlist_size());
}
ArrayRef<const Expr *> getDestinationExprs() const {
return llvm::makeArrayRef(getSourceExprs().end(), varlist_size());
}

void setAssignmentOps(ArrayRef<Expr *> AssignmentOps);

MutableArrayRef<Expr *> getAssignmentOps() {
return MutableArrayRef<Expr *>(getDestinationExprs().end(), varlist_size());
}
ArrayRef<const Expr *> getAssignmentOps() const {
return llvm::makeArrayRef(getDestinationExprs().end(), varlist_size());
}

public:
static OMPCopyinClause *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc, ArrayRef<Expr *> VL, ArrayRef<Expr *> SrcExprs,
ArrayRef<Expr *> DstExprs, ArrayRef<Expr *> AssignmentOps);

static OMPCopyinClause *CreateEmpty(const ASTContext &C, unsigned N);

using helper_expr_iterator = MutableArrayRef<Expr *>::iterator;
using helper_expr_const_iterator = ArrayRef<const Expr *>::iterator;
using helper_expr_range = llvm::iterator_range<helper_expr_iterator>;
using helper_expr_const_range =
llvm::iterator_range<helper_expr_const_iterator>;

helper_expr_const_range source_exprs() const {
return helper_expr_const_range(getSourceExprs().begin(),
getSourceExprs().end());
}

helper_expr_range source_exprs() {
return helper_expr_range(getSourceExprs().begin(), getSourceExprs().end());
}

helper_expr_const_range destination_exprs() const {
return helper_expr_const_range(getDestinationExprs().begin(),
getDestinationExprs().end());
}

helper_expr_range destination_exprs() {
return helper_expr_range(getDestinationExprs().begin(),
getDestinationExprs().end());
}

helper_expr_const_range assignment_ops() const {
return helper_expr_const_range(getAssignmentOps().begin(),
getAssignmentOps().end());
}

helper_expr_range assignment_ops() {
return helper_expr_range(getAssignmentOps().begin(),
getAssignmentOps().end());
}

child_range children() {
return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
reinterpret_cast<Stmt **>(varlist_end()));
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_copyin;
}
};

class OMPCopyprivateClause final
: public OMPVarListClause<OMPCopyprivateClause>,
private llvm::TrailingObjects<OMPCopyprivateClause, Expr *> {
friend class OMPClauseReader;
friend OMPVarListClause;
friend TrailingObjects;

OMPCopyprivateClause(SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc, unsigned N)
: OMPVarListClause<OMPCopyprivateClause>(OMPC_copyprivate, StartLoc,
LParenLoc, EndLoc, N) {}

explicit OMPCopyprivateClause(unsigned N)
: OMPVarListClause<OMPCopyprivateClause>(
OMPC_copyprivate, SourceLocation(), SourceLocation(),
SourceLocation(), N) {}

void setSourceExprs(ArrayRef<Expr *> SrcExprs);

MutableArrayRef<Expr *> getSourceExprs() {
return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
}
ArrayRef<const Expr *> getSourceExprs() const {
return llvm::makeArrayRef(varlist_end(), varlist_size());
}

void setDestinationExprs(ArrayRef<Expr *> DstExprs);

MutableArrayRef<Expr *> getDestinationExprs() {
return MutableArrayRef<Expr *>(getSourceExprs().end(), varlist_size());
}
ArrayRef<const Expr *> getDestinationExprs() const {
return llvm::makeArrayRef(getSourceExprs().end(), varlist_size());
}

void setAssignmentOps(ArrayRef<Expr *> AssignmentOps);

MutableArrayRef<Expr *> getAssignmentOps() {
return MutableArrayRef<Expr *>(getDestinationExprs().end(), varlist_size());
}
ArrayRef<const Expr *> getAssignmentOps() const {
return llvm::makeArrayRef(getDestinationExprs().end(), varlist_size());
}

public:
static OMPCopyprivateClause *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc, ArrayRef<Expr *> VL, ArrayRef<Expr *> SrcExprs,
ArrayRef<Expr *> DstExprs, ArrayRef<Expr *> AssignmentOps);

static OMPCopyprivateClause *CreateEmpty(const ASTContext &C, unsigned N);

using helper_expr_iterator = MutableArrayRef<Expr *>::iterator;
using helper_expr_const_iterator = ArrayRef<const Expr *>::iterator;
using helper_expr_range = llvm::iterator_range<helper_expr_iterator>;
using helper_expr_const_range =
llvm::iterator_range<helper_expr_const_iterator>;

helper_expr_const_range source_exprs() const {
return helper_expr_const_range(getSourceExprs().begin(),
getSourceExprs().end());
}

helper_expr_range source_exprs() {
return helper_expr_range(getSourceExprs().begin(), getSourceExprs().end());
}

helper_expr_const_range destination_exprs() const {
return helper_expr_const_range(getDestinationExprs().begin(),
getDestinationExprs().end());
}

helper_expr_range destination_exprs() {
return helper_expr_range(getDestinationExprs().begin(),
getDestinationExprs().end());
}

helper_expr_const_range assignment_ops() const {
return helper_expr_const_range(getAssignmentOps().begin(),
getAssignmentOps().end());
}

helper_expr_range assignment_ops() {
return helper_expr_range(getAssignmentOps().begin(),
getAssignmentOps().end());
}

child_range children() {
return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
reinterpret_cast<Stmt **>(varlist_end()));
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_copyprivate;
}
};

class OMPFlushClause final
: public OMPVarListClause<OMPFlushClause>,
private llvm::TrailingObjects<OMPFlushClause, Expr *> {
friend OMPVarListClause;
friend TrailingObjects;

OMPFlushClause(SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc, unsigned N)
: OMPVarListClause<OMPFlushClause>(OMPC_flush, StartLoc, LParenLoc,
EndLoc, N) {}

explicit OMPFlushClause(unsigned N)
: OMPVarListClause<OMPFlushClause>(OMPC_flush, SourceLocation(),
SourceLocation(), SourceLocation(),
N) {}

public:
static OMPFlushClause *Create(const ASTContext &C, SourceLocation StartLoc,
SourceLocation LParenLoc, SourceLocation EndLoc,
ArrayRef<Expr *> VL);

static OMPFlushClause *CreateEmpty(const ASTContext &C, unsigned N);

child_range children() {
return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
reinterpret_cast<Stmt **>(varlist_end()));
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_flush;
}
};

class OMPDependClause final
: public OMPVarListClause<OMPDependClause>,
private llvm::TrailingObjects<OMPDependClause, Expr *> {
friend class OMPClauseReader;
friend OMPVarListClause;
friend TrailingObjects;

OpenMPDependClauseKind DepKind = OMPC_DEPEND_unknown;

SourceLocation DepLoc;

SourceLocation ColonLoc;

unsigned NumLoops = 0;

OMPDependClause(SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc, unsigned N, unsigned NumLoops)
: OMPVarListClause<OMPDependClause>(OMPC_depend, StartLoc, LParenLoc,
EndLoc, N), NumLoops(NumLoops) {}

explicit OMPDependClause(unsigned N, unsigned NumLoops)
: OMPVarListClause<OMPDependClause>(OMPC_depend, SourceLocation(),
SourceLocation(), SourceLocation(),
N),
NumLoops(NumLoops) {}

void setDependencyKind(OpenMPDependClauseKind K) { DepKind = K; }

void setDependencyLoc(SourceLocation Loc) { DepLoc = Loc; }

void setColonLoc(SourceLocation Loc) { ColonLoc = Loc; }

public:
static OMPDependClause *Create(const ASTContext &C, SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc,
OpenMPDependClauseKind DepKind,
SourceLocation DepLoc, SourceLocation ColonLoc,
ArrayRef<Expr *> VL, unsigned NumLoops);

static OMPDependClause *CreateEmpty(const ASTContext &C, unsigned N,
unsigned NumLoops);

OpenMPDependClauseKind getDependencyKind() const { return DepKind; }

SourceLocation getDependencyLoc() const { return DepLoc; }

SourceLocation getColonLoc() const { return ColonLoc; }

unsigned getNumLoops() const { return NumLoops; }

void setLoopData(unsigned NumLoop, Expr *Cnt);

Expr *getLoopData(unsigned NumLoop);
const Expr *getLoopData(unsigned NumLoop) const;

child_range children() {
return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
reinterpret_cast<Stmt **>(varlist_end()));
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_depend;
}
};

class OMPDeviceClause : public OMPClause, public OMPClauseWithPreInit {
friend class OMPClauseReader;

SourceLocation LParenLoc;

Stmt *Device = nullptr;

void setDevice(Expr *E) { Device = E; }

public:
OMPDeviceClause(Expr *E, Stmt *HelperE, OpenMPDirectiveKind CaptureRegion,
SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc)
: OMPClause(OMPC_device, StartLoc, EndLoc), OMPClauseWithPreInit(this),
LParenLoc(LParenLoc), Device(E) {
setPreInitStmt(HelperE, CaptureRegion);
}

OMPDeviceClause()
: OMPClause(OMPC_device, SourceLocation(), SourceLocation()),
OMPClauseWithPreInit(this) {}

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

SourceLocation getLParenLoc() const { return LParenLoc; }

Expr *getDevice() { return cast<Expr>(Device); }

Expr *getDevice() const { return cast<Expr>(Device); }

child_range children() { return child_range(&Device, &Device + 1); }

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_device;
}
};

class OMPThreadsClause : public OMPClause {
public:
OMPThreadsClause(SourceLocation StartLoc, SourceLocation EndLoc)
: OMPClause(OMPC_threads, StartLoc, EndLoc) {}

OMPThreadsClause()
: OMPClause(OMPC_threads, SourceLocation(), SourceLocation()) {}

child_range children() {
return child_range(child_iterator(), child_iterator());
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_threads;
}
};

class OMPSIMDClause : public OMPClause {
public:
OMPSIMDClause(SourceLocation StartLoc, SourceLocation EndLoc)
: OMPClause(OMPC_simd, StartLoc, EndLoc) {}

OMPSIMDClause() : OMPClause(OMPC_simd, SourceLocation(), SourceLocation()) {}

child_range children() {
return child_range(child_iterator(), child_iterator());
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_simd;
}
};

class OMPClauseMappableExprCommon {
public:
class MappableComponent {
Expr *AssociatedExpression = nullptr;

ValueDecl *AssociatedDeclaration = nullptr;

public:
explicit MappableComponent() = default;
explicit MappableComponent(Expr *AssociatedExpression,
ValueDecl *AssociatedDeclaration)
: AssociatedExpression(AssociatedExpression),
AssociatedDeclaration(
AssociatedDeclaration
? cast<ValueDecl>(AssociatedDeclaration->getCanonicalDecl())
: nullptr) {}

Expr *getAssociatedExpression() const { return AssociatedExpression; }

ValueDecl *getAssociatedDeclaration() const {
return AssociatedDeclaration;
}
};

using MappableExprComponentList = SmallVector<MappableComponent, 8>;
using MappableExprComponentListRef = ArrayRef<MappableComponent>;

using MappableExprComponentLists = SmallVector<MappableExprComponentList, 8>;
using MappableExprComponentListsRef = ArrayRef<MappableExprComponentList>;

protected:
static unsigned
getComponentsTotalNumber(MappableExprComponentListsRef ComponentLists);

static unsigned
getUniqueDeclarationsTotalNumber(ArrayRef<const ValueDecl *> Declarations);
};

template <class T>
class OMPMappableExprListClause : public OMPVarListClause<T>,
public OMPClauseMappableExprCommon {
friend class OMPClauseReader;

unsigned NumUniqueDeclarations;

unsigned NumComponentLists;

unsigned NumComponents;

protected:
OMPMappableExprListClause(OpenMPClauseKind K, SourceLocation StartLoc,
SourceLocation LParenLoc, SourceLocation EndLoc,
unsigned NumVars, unsigned NumUniqueDeclarations,
unsigned NumComponentLists, unsigned NumComponents)
: OMPVarListClause<T>(K, StartLoc, LParenLoc, EndLoc, NumVars),
NumUniqueDeclarations(NumUniqueDeclarations),
NumComponentLists(NumComponentLists), NumComponents(NumComponents) {}

MutableArrayRef<ValueDecl *> getUniqueDeclsRef() {
return MutableArrayRef<ValueDecl *>(
static_cast<T *>(this)->template getTrailingObjects<ValueDecl *>(),
NumUniqueDeclarations);
}

ArrayRef<ValueDecl *> getUniqueDeclsRef() const {
return ArrayRef<ValueDecl *>(
static_cast<const T *>(this)
->template getTrailingObjects<ValueDecl *>(),
NumUniqueDeclarations);
}

void setUniqueDecls(ArrayRef<ValueDecl *> UDs) {
assert(UDs.size() == NumUniqueDeclarations &&
"Unexpected amount of unique declarations.");
std::copy(UDs.begin(), UDs.end(), getUniqueDeclsRef().begin());
}

MutableArrayRef<unsigned> getDeclNumListsRef() {
return MutableArrayRef<unsigned>(
static_cast<T *>(this)->template getTrailingObjects<unsigned>(),
NumUniqueDeclarations);
}

ArrayRef<unsigned> getDeclNumListsRef() const {
return ArrayRef<unsigned>(
static_cast<const T *>(this)->template getTrailingObjects<unsigned>(),
NumUniqueDeclarations);
}

void setDeclNumLists(ArrayRef<unsigned> DNLs) {
assert(DNLs.size() == NumUniqueDeclarations &&
"Unexpected amount of list numbers.");
std::copy(DNLs.begin(), DNLs.end(), getDeclNumListsRef().begin());
}

MutableArrayRef<unsigned> getComponentListSizesRef() {
return MutableArrayRef<unsigned>(
static_cast<T *>(this)->template getTrailingObjects<unsigned>() +
NumUniqueDeclarations,
NumComponentLists);
}

ArrayRef<unsigned> getComponentListSizesRef() const {
return ArrayRef<unsigned>(
static_cast<const T *>(this)->template getTrailingObjects<unsigned>() +
NumUniqueDeclarations,
NumComponentLists);
}

void setComponentListSizes(ArrayRef<unsigned> CLSs) {
assert(CLSs.size() == NumComponentLists &&
"Unexpected amount of component lists.");
std::copy(CLSs.begin(), CLSs.end(), getComponentListSizesRef().begin());
}

MutableArrayRef<MappableComponent> getComponentsRef() {
return MutableArrayRef<MappableComponent>(
static_cast<T *>(this)
->template getTrailingObjects<MappableComponent>(),
NumComponents);
}

ArrayRef<MappableComponent> getComponentsRef() const {
return ArrayRef<MappableComponent>(
static_cast<const T *>(this)
->template getTrailingObjects<MappableComponent>(),
NumComponents);
}

void setComponents(ArrayRef<MappableComponent> Components,
ArrayRef<unsigned> CLSs) {
assert(Components.size() == NumComponents &&
"Unexpected amount of component lists.");
assert(CLSs.size() == NumComponentLists &&
"Unexpected amount of list sizes.");
std::copy(Components.begin(), Components.end(), getComponentsRef().begin());
}

void setClauseInfo(ArrayRef<ValueDecl *> Declarations,
MappableExprComponentListsRef ComponentLists) {
assert(getUniqueDeclarationsTotalNumber(Declarations) ==
NumUniqueDeclarations &&
"Unexpected number of mappable expression info entries!");
assert(getComponentsTotalNumber(ComponentLists) == NumComponents &&
"Unexpected total number of components!");
assert(Declarations.size() == ComponentLists.size() &&
"Declaration and component lists size is not consistent!");
assert(Declarations.size() == NumComponentLists &&
"Unexpected declaration and component lists size!");

llvm::MapVector<ValueDecl *, SmallVector<MappableExprComponentListRef, 8>>
ComponentListMap;
{
auto CI = ComponentLists.begin();
for (auto DI = Declarations.begin(), DE = Declarations.end(); DI != DE;
++DI, ++CI) {
assert(!CI->empty() && "Invalid component list!");
ComponentListMap[*DI].push_back(*CI);
}
}

auto UniqueDeclarations = getUniqueDeclsRef();
auto UDI = UniqueDeclarations.begin();

auto DeclNumLists = getDeclNumListsRef();
auto DNLI = DeclNumLists.begin();

auto ComponentListSizes = getComponentListSizesRef();
auto CLSI = ComponentListSizes.begin();

auto Components = getComponentsRef();
auto CI = Components.begin();

unsigned PrevSize = 0u;

for (auto &M : ComponentListMap) {
auto *D = M.first;
auto CL = M.second;

*UDI = D;
++UDI;

*DNLI = CL.size();
++DNLI;

for (auto C : CL) {
PrevSize += C.size();

*CLSI = PrevSize;
++CLSI;

CI = std::copy(C.begin(), C.end(), CI);
}
}
}

public:
unsigned getUniqueDeclarationsNum() const { return NumUniqueDeclarations; }

unsigned getTotalComponentListNum() const { return NumComponentLists; }

unsigned getTotalComponentsNum() const { return NumComponents; }

class const_component_lists_iterator
: public llvm::iterator_adaptor_base<
const_component_lists_iterator,
MappableExprComponentListRef::const_iterator,
std::forward_iterator_tag, MappableComponent, ptrdiff_t,
MappableComponent, MappableComponent> {
ArrayRef<ValueDecl *>::iterator DeclCur;

ArrayRef<unsigned>::iterator NumListsCur;

unsigned RemainingLists = 0;

unsigned PrevListSize = 0;

ArrayRef<unsigned>::const_iterator ListSizeCur;
ArrayRef<unsigned>::const_iterator ListSizeEnd;

MappableExprComponentListRef::const_iterator End;

public:
explicit const_component_lists_iterator(
ArrayRef<ValueDecl *> UniqueDecls, ArrayRef<unsigned> DeclsListNum,
ArrayRef<unsigned> CumulativeListSizes,
MappableExprComponentListRef Components)
: const_component_lists_iterator::iterator_adaptor_base(
Components.begin()),
DeclCur(UniqueDecls.begin()), NumListsCur(DeclsListNum.begin()),
ListSizeCur(CumulativeListSizes.begin()),
ListSizeEnd(CumulativeListSizes.end()), End(Components.end()) {
assert(UniqueDecls.size() == DeclsListNum.size() &&
"Inconsistent number of declarations and list sizes!");
if (!DeclsListNum.empty())
RemainingLists = *NumListsCur;
}

explicit const_component_lists_iterator(
const ValueDecl *Declaration, ArrayRef<ValueDecl *> UniqueDecls,
ArrayRef<unsigned> DeclsListNum, ArrayRef<unsigned> CumulativeListSizes,
MappableExprComponentListRef Components)
: const_component_lists_iterator(UniqueDecls, DeclsListNum,
CumulativeListSizes, Components) {
for (; DeclCur != UniqueDecls.end(); ++DeclCur, ++NumListsCur) {
if (*DeclCur == Declaration)
break;

assert(*NumListsCur > 0 && "No lists associated with declaration??");

std::advance(ListSizeCur, *NumListsCur - 1);
PrevListSize = *ListSizeCur;
++ListSizeCur;
}

if (ListSizeCur == CumulativeListSizes.end()) {
this->I = End;
RemainingLists = 0u;
return;
}

RemainingLists = *NumListsCur;

ListSizeEnd = ListSizeCur;
std::advance(ListSizeEnd, RemainingLists);

std::advance(this->I, PrevListSize);
}

std::pair<const ValueDecl *, MappableExprComponentListRef>
operator*() const {
assert(ListSizeCur != ListSizeEnd && "Invalid iterator!");
return std::make_pair(
*DeclCur,
MappableExprComponentListRef(&*this->I, *ListSizeCur - PrevListSize));
}
std::pair<const ValueDecl *, MappableExprComponentListRef>
operator->() const {
return **this;
}

const_component_lists_iterator &operator++() {
assert(ListSizeCur != ListSizeEnd && RemainingLists &&
"Invalid iterator!");

if (std::next(ListSizeCur) == ListSizeEnd) {
this->I = End;
RemainingLists = 0;
} else {
std::advance(this->I, *ListSizeCur - PrevListSize);
PrevListSize = *ListSizeCur;

if (!(--RemainingLists)) {
++DeclCur;
++NumListsCur;
RemainingLists = *NumListsCur;
assert(RemainingLists && "No lists in the following declaration??");
}
}

++ListSizeCur;
return *this;
}
};

using const_component_lists_range =
llvm::iterator_range<const_component_lists_iterator>;

const_component_lists_iterator component_lists_begin() const {
return const_component_lists_iterator(
getUniqueDeclsRef(), getDeclNumListsRef(), getComponentListSizesRef(),
getComponentsRef());
}
const_component_lists_iterator component_lists_end() const {
return const_component_lists_iterator(
ArrayRef<ValueDecl *>(), ArrayRef<unsigned>(), ArrayRef<unsigned>(),
MappableExprComponentListRef(getComponentsRef().end(),
getComponentsRef().end()));
}
const_component_lists_range component_lists() const {
return {component_lists_begin(), component_lists_end()};
}

const_component_lists_iterator
decl_component_lists_begin(const ValueDecl *VD) const {
return const_component_lists_iterator(
VD, getUniqueDeclsRef(), getDeclNumListsRef(),
getComponentListSizesRef(), getComponentsRef());
}
const_component_lists_iterator decl_component_lists_end() const {
return component_lists_end();
}
const_component_lists_range decl_component_lists(const ValueDecl *VD) const {
return {decl_component_lists_begin(VD), decl_component_lists_end()};
}

using const_all_decls_iterator = ArrayRef<ValueDecl *>::iterator;
using const_all_decls_range = llvm::iterator_range<const_all_decls_iterator>;

const_all_decls_range all_decls() const {
auto A = getUniqueDeclsRef();
return const_all_decls_range(A.begin(), A.end());
}

using const_all_num_lists_iterator = ArrayRef<unsigned>::iterator;
using const_all_num_lists_range =
llvm::iterator_range<const_all_num_lists_iterator>;

const_all_num_lists_range all_num_lists() const {
auto A = getDeclNumListsRef();
return const_all_num_lists_range(A.begin(), A.end());
}

using const_all_lists_sizes_iterator = ArrayRef<unsigned>::iterator;
using const_all_lists_sizes_range =
llvm::iterator_range<const_all_lists_sizes_iterator>;

const_all_lists_sizes_range all_lists_sizes() const {
auto A = getComponentListSizesRef();
return const_all_lists_sizes_range(A.begin(), A.end());
}

using const_all_components_iterator = ArrayRef<MappableComponent>::iterator;
using const_all_components_range =
llvm::iterator_range<const_all_components_iterator>;

const_all_components_range all_components() const {
auto A = getComponentsRef();
return const_all_components_range(A.begin(), A.end());
}
};

class OMPMapClause final : public OMPMappableExprListClause<OMPMapClause>,
private llvm::TrailingObjects<
OMPMapClause, Expr *, ValueDecl *, unsigned,
OMPClauseMappableExprCommon::MappableComponent> {
friend class OMPClauseReader;
friend OMPMappableExprListClause;
friend OMPVarListClause;
friend TrailingObjects;

size_t numTrailingObjects(OverloadToken<Expr *>) const {
return varlist_size();
}
size_t numTrailingObjects(OverloadToken<ValueDecl *>) const {
return getUniqueDeclarationsNum();
}
size_t numTrailingObjects(OverloadToken<unsigned>) const {
return getUniqueDeclarationsNum() + getTotalComponentListNum();
}

OpenMPMapClauseKind MapTypeModifier = OMPC_MAP_unknown;

OpenMPMapClauseKind MapType = OMPC_MAP_unknown;

bool MapTypeIsImplicit = false;

SourceLocation MapLoc;

SourceLocation ColonLoc;

explicit OMPMapClause(OpenMPMapClauseKind MapTypeModifier,
OpenMPMapClauseKind MapType, bool MapTypeIsImplicit,
SourceLocation MapLoc, SourceLocation StartLoc,
SourceLocation LParenLoc, SourceLocation EndLoc,
unsigned NumVars, unsigned NumUniqueDeclarations,
unsigned NumComponentLists, unsigned NumComponents)
: OMPMappableExprListClause(OMPC_map, StartLoc, LParenLoc, EndLoc,
NumVars, NumUniqueDeclarations,
NumComponentLists, NumComponents),
MapTypeModifier(MapTypeModifier), MapType(MapType),
MapTypeIsImplicit(MapTypeIsImplicit), MapLoc(MapLoc) {}

explicit OMPMapClause(unsigned NumVars, unsigned NumUniqueDeclarations,
unsigned NumComponentLists, unsigned NumComponents)
: OMPMappableExprListClause(
OMPC_map, SourceLocation(), SourceLocation(), SourceLocation(),
NumVars, NumUniqueDeclarations, NumComponentLists, NumComponents) {}

void setMapTypeModifier(OpenMPMapClauseKind T) { MapTypeModifier = T; }

void setMapType(OpenMPMapClauseKind T) { MapType = T; }

void setMapLoc(SourceLocation TLoc) { MapLoc = TLoc; }

void setColonLoc(SourceLocation Loc) { ColonLoc = Loc; }

public:
static OMPMapClause *Create(const ASTContext &C, SourceLocation StartLoc,
SourceLocation LParenLoc, SourceLocation EndLoc,
ArrayRef<Expr *> Vars,
ArrayRef<ValueDecl *> Declarations,
MappableExprComponentListsRef ComponentLists,
OpenMPMapClauseKind TypeModifier,
OpenMPMapClauseKind Type, bool TypeIsImplicit,
SourceLocation TypeLoc);

static OMPMapClause *CreateEmpty(const ASTContext &C, unsigned NumVars,
unsigned NumUniqueDeclarations,
unsigned NumComponentLists,
unsigned NumComponents);

OpenMPMapClauseKind getMapType() const LLVM_READONLY { return MapType; }

bool isImplicitMapType() const LLVM_READONLY { return MapTypeIsImplicit; }

OpenMPMapClauseKind getMapTypeModifier() const LLVM_READONLY {
return MapTypeModifier;
}

SourceLocation getMapLoc() const LLVM_READONLY { return MapLoc; }

SourceLocation getColonLoc() const { return ColonLoc; }

child_range children() {
return child_range(
reinterpret_cast<Stmt **>(varlist_begin()),
reinterpret_cast<Stmt **>(varlist_end()));
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_map;
}
};

class OMPNumTeamsClause : public OMPClause, public OMPClauseWithPreInit {
friend class OMPClauseReader;

SourceLocation LParenLoc;

Stmt *NumTeams = nullptr;

void setNumTeams(Expr *E) { NumTeams = E; }

public:
OMPNumTeamsClause(Expr *E, Stmt *HelperE, OpenMPDirectiveKind CaptureRegion,
SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc)
: OMPClause(OMPC_num_teams, StartLoc, EndLoc), OMPClauseWithPreInit(this),
LParenLoc(LParenLoc), NumTeams(E) {
setPreInitStmt(HelperE, CaptureRegion);
}

OMPNumTeamsClause()
: OMPClause(OMPC_num_teams, SourceLocation(), SourceLocation()),
OMPClauseWithPreInit(this) {}

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

SourceLocation getLParenLoc() const { return LParenLoc; }

Expr *getNumTeams() { return cast<Expr>(NumTeams); }

Expr *getNumTeams() const { return cast<Expr>(NumTeams); }

child_range children() { return child_range(&NumTeams, &NumTeams + 1); }

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_num_teams;
}
};

class OMPThreadLimitClause : public OMPClause, public OMPClauseWithPreInit {
friend class OMPClauseReader;

SourceLocation LParenLoc;

Stmt *ThreadLimit = nullptr;

void setThreadLimit(Expr *E) { ThreadLimit = E; }

public:
OMPThreadLimitClause(Expr *E, Stmt *HelperE,
OpenMPDirectiveKind CaptureRegion,
SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc)
: OMPClause(OMPC_thread_limit, StartLoc, EndLoc),
OMPClauseWithPreInit(this), LParenLoc(LParenLoc), ThreadLimit(E) {
setPreInitStmt(HelperE, CaptureRegion);
}

OMPThreadLimitClause()
: OMPClause(OMPC_thread_limit, SourceLocation(), SourceLocation()),
OMPClauseWithPreInit(this) {}

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

SourceLocation getLParenLoc() const { return LParenLoc; }

Expr *getThreadLimit() { return cast<Expr>(ThreadLimit); }

Expr *getThreadLimit() const { return cast<Expr>(ThreadLimit); }

child_range children() { return child_range(&ThreadLimit, &ThreadLimit + 1); }

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_thread_limit;
}
};

class OMPPriorityClause : public OMPClause {
friend class OMPClauseReader;

SourceLocation LParenLoc;

Stmt *Priority = nullptr;

void setPriority(Expr *E) { Priority = E; }

public:
OMPPriorityClause(Expr *E, SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc)
: OMPClause(OMPC_priority, StartLoc, EndLoc), LParenLoc(LParenLoc),
Priority(E) {}

OMPPriorityClause()
: OMPClause(OMPC_priority, SourceLocation(), SourceLocation()) {}

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

SourceLocation getLParenLoc() const { return LParenLoc; }

Expr *getPriority() { return cast<Expr>(Priority); }

Expr *getPriority() const { return cast<Expr>(Priority); }

child_range children() { return child_range(&Priority, &Priority + 1); }

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_priority;
}
};

class OMPGrainsizeClause : public OMPClause {
friend class OMPClauseReader;

SourceLocation LParenLoc;

Stmt *Grainsize = nullptr;

void setGrainsize(Expr *Size) { Grainsize = Size; }

public:
OMPGrainsizeClause(Expr *Size, SourceLocation StartLoc,
SourceLocation LParenLoc, SourceLocation EndLoc)
: OMPClause(OMPC_grainsize, StartLoc, EndLoc), LParenLoc(LParenLoc),
Grainsize(Size) {}

explicit OMPGrainsizeClause()
: OMPClause(OMPC_grainsize, SourceLocation(), SourceLocation()) {}

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

SourceLocation getLParenLoc() const { return LParenLoc; }

Expr *getGrainsize() const { return cast_or_null<Expr>(Grainsize); }

child_range children() { return child_range(&Grainsize, &Grainsize + 1); }

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_grainsize;
}
};

class OMPNogroupClause : public OMPClause {
public:
OMPNogroupClause(SourceLocation StartLoc, SourceLocation EndLoc)
: OMPClause(OMPC_nogroup, StartLoc, EndLoc) {}

OMPNogroupClause()
: OMPClause(OMPC_nogroup, SourceLocation(), SourceLocation()) {}

child_range children() {
return child_range(child_iterator(), child_iterator());
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_nogroup;
}
};

class OMPNumTasksClause : public OMPClause {
friend class OMPClauseReader;

SourceLocation LParenLoc;

Stmt *NumTasks = nullptr;

void setNumTasks(Expr *Size) { NumTasks = Size; }

public:
OMPNumTasksClause(Expr *Size, SourceLocation StartLoc,
SourceLocation LParenLoc, SourceLocation EndLoc)
: OMPClause(OMPC_num_tasks, StartLoc, EndLoc), LParenLoc(LParenLoc),
NumTasks(Size) {}

explicit OMPNumTasksClause()
: OMPClause(OMPC_num_tasks, SourceLocation(), SourceLocation()) {}

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

SourceLocation getLParenLoc() const { return LParenLoc; }

Expr *getNumTasks() const { return cast_or_null<Expr>(NumTasks); }

child_range children() { return child_range(&NumTasks, &NumTasks + 1); }

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_num_tasks;
}
};

class OMPHintClause : public OMPClause {
friend class OMPClauseReader;

SourceLocation LParenLoc;

Stmt *Hint = nullptr;

void setHint(Expr *H) { Hint = H; }

public:
OMPHintClause(Expr *Hint, SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc)
: OMPClause(OMPC_hint, StartLoc, EndLoc), LParenLoc(LParenLoc),
Hint(Hint) {}

OMPHintClause() : OMPClause(OMPC_hint, SourceLocation(), SourceLocation()) {}

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

SourceLocation getLParenLoc() const { return LParenLoc; }

Expr *getHint() const { return cast_or_null<Expr>(Hint); }

child_range children() { return child_range(&Hint, &Hint + 1); }

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_hint;
}
};

class OMPDistScheduleClause : public OMPClause, public OMPClauseWithPreInit {
friend class OMPClauseReader;

SourceLocation LParenLoc;

OpenMPDistScheduleClauseKind Kind = OMPC_DIST_SCHEDULE_unknown;

SourceLocation KindLoc;

SourceLocation CommaLoc;

Expr *ChunkSize = nullptr;

void setDistScheduleKind(OpenMPDistScheduleClauseKind K) { Kind = K; }

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

void setDistScheduleKindLoc(SourceLocation KLoc) { KindLoc = KLoc; }

void setCommaLoc(SourceLocation Loc) { CommaLoc = Loc; }

void setChunkSize(Expr *E) { ChunkSize = E; }

public:
OMPDistScheduleClause(SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation KLoc, SourceLocation CommaLoc,
SourceLocation EndLoc,
OpenMPDistScheduleClauseKind Kind, Expr *ChunkSize,
Stmt *HelperChunkSize)
: OMPClause(OMPC_dist_schedule, StartLoc, EndLoc),
OMPClauseWithPreInit(this), LParenLoc(LParenLoc), Kind(Kind),
KindLoc(KLoc), CommaLoc(CommaLoc), ChunkSize(ChunkSize) {
setPreInitStmt(HelperChunkSize);
}

explicit OMPDistScheduleClause()
: OMPClause(OMPC_dist_schedule, SourceLocation(), SourceLocation()),
OMPClauseWithPreInit(this) {}

OpenMPDistScheduleClauseKind getDistScheduleKind() const { return Kind; }

SourceLocation getLParenLoc() { return LParenLoc; }

SourceLocation getDistScheduleKindLoc() { return KindLoc; }

SourceLocation getCommaLoc() { return CommaLoc; }

Expr *getChunkSize() { return ChunkSize; }

const Expr *getChunkSize() const { return ChunkSize; }

child_range children() {
return child_range(reinterpret_cast<Stmt **>(&ChunkSize),
reinterpret_cast<Stmt **>(&ChunkSize) + 1);
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_dist_schedule;
}
};

class OMPDefaultmapClause : public OMPClause {
friend class OMPClauseReader;

SourceLocation LParenLoc;

OpenMPDefaultmapClauseModifier Modifier = OMPC_DEFAULTMAP_MODIFIER_unknown;

SourceLocation ModifierLoc;

OpenMPDefaultmapClauseKind Kind = OMPC_DEFAULTMAP_unknown;

SourceLocation KindLoc;

void setDefaultmapKind(OpenMPDefaultmapClauseKind K) { Kind = K; }

void setDefaultmapModifier(OpenMPDefaultmapClauseModifier M) {
Modifier = M;
}

void setDefaultmapModifierLoc(SourceLocation Loc) {
ModifierLoc = Loc;
}

void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

void setDefaultmapKindLoc(SourceLocation KLoc) { KindLoc = KLoc; }

public:
OMPDefaultmapClause(SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation MLoc, SourceLocation KLoc,
SourceLocation EndLoc, OpenMPDefaultmapClauseKind Kind,
OpenMPDefaultmapClauseModifier M)
: OMPClause(OMPC_defaultmap, StartLoc, EndLoc), LParenLoc(LParenLoc),
Modifier(M), ModifierLoc(MLoc), Kind(Kind), KindLoc(KLoc) {}

explicit OMPDefaultmapClause()
: OMPClause(OMPC_defaultmap, SourceLocation(), SourceLocation()) {}

OpenMPDefaultmapClauseKind getDefaultmapKind() const { return Kind; }

OpenMPDefaultmapClauseModifier getDefaultmapModifier() const {
return Modifier;
}

SourceLocation getLParenLoc() { return LParenLoc; }

SourceLocation getDefaultmapKindLoc() { return KindLoc; }

SourceLocation getDefaultmapModifierLoc() const {
return ModifierLoc;
}

child_range children() {
return child_range(child_iterator(), child_iterator());
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_defaultmap;
}
};

class OMPToClause final : public OMPMappableExprListClause<OMPToClause>,
private llvm::TrailingObjects<
OMPToClause, Expr *, ValueDecl *, unsigned,
OMPClauseMappableExprCommon::MappableComponent> {
friend class OMPClauseReader;
friend OMPMappableExprListClause;
friend OMPVarListClause;
friend TrailingObjects;

explicit OMPToClause(SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc, unsigned NumVars,
unsigned NumUniqueDeclarations,
unsigned NumComponentLists, unsigned NumComponents)
: OMPMappableExprListClause(OMPC_to, StartLoc, LParenLoc, EndLoc, NumVars,
NumUniqueDeclarations, NumComponentLists,
NumComponents) {}

explicit OMPToClause(unsigned NumVars, unsigned NumUniqueDeclarations,
unsigned NumComponentLists, unsigned NumComponents)
: OMPMappableExprListClause(
OMPC_to, SourceLocation(), SourceLocation(), SourceLocation(),
NumVars, NumUniqueDeclarations, NumComponentLists, NumComponents) {}

size_t numTrailingObjects(OverloadToken<Expr *>) const {
return varlist_size();
}
size_t numTrailingObjects(OverloadToken<ValueDecl *>) const {
return getUniqueDeclarationsNum();
}
size_t numTrailingObjects(OverloadToken<unsigned>) const {
return getUniqueDeclarationsNum() + getTotalComponentListNum();
}

public:
static OMPToClause *Create(const ASTContext &C, SourceLocation StartLoc,
SourceLocation LParenLoc, SourceLocation EndLoc,
ArrayRef<Expr *> Vars,
ArrayRef<ValueDecl *> Declarations,
MappableExprComponentListsRef ComponentLists);

static OMPToClause *CreateEmpty(const ASTContext &C, unsigned NumVars,
unsigned NumUniqueDeclarations,
unsigned NumComponentLists,
unsigned NumComponents);

child_range children() {
return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
reinterpret_cast<Stmt **>(varlist_end()));
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_to;
}
};

class OMPFromClause final
: public OMPMappableExprListClause<OMPFromClause>,
private llvm::TrailingObjects<
OMPFromClause, Expr *, ValueDecl *, unsigned,
OMPClauseMappableExprCommon::MappableComponent> {
friend class OMPClauseReader;
friend OMPMappableExprListClause;
friend OMPVarListClause;
friend TrailingObjects;

explicit OMPFromClause(SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc, unsigned NumVars,
unsigned NumUniqueDeclarations,
unsigned NumComponentLists, unsigned NumComponents)
: OMPMappableExprListClause(OMPC_from, StartLoc, LParenLoc, EndLoc,
NumVars, NumUniqueDeclarations,
NumComponentLists, NumComponents) {}

explicit OMPFromClause(unsigned NumVars, unsigned NumUniqueDeclarations,
unsigned NumComponentLists, unsigned NumComponents)
: OMPMappableExprListClause(
OMPC_from, SourceLocation(), SourceLocation(), SourceLocation(),
NumVars, NumUniqueDeclarations, NumComponentLists, NumComponents) {}

size_t numTrailingObjects(OverloadToken<Expr *>) const {
return varlist_size();
}
size_t numTrailingObjects(OverloadToken<ValueDecl *>) const {
return getUniqueDeclarationsNum();
}
size_t numTrailingObjects(OverloadToken<unsigned>) const {
return getUniqueDeclarationsNum() + getTotalComponentListNum();
}

public:
static OMPFromClause *Create(const ASTContext &C, SourceLocation StartLoc,
SourceLocation LParenLoc, SourceLocation EndLoc,
ArrayRef<Expr *> Vars,
ArrayRef<ValueDecl *> Declarations,
MappableExprComponentListsRef ComponentLists);

static OMPFromClause *CreateEmpty(const ASTContext &C, unsigned NumVars,
unsigned NumUniqueDeclarations,
unsigned NumComponentLists,
unsigned NumComponents);

child_range children() {
return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
reinterpret_cast<Stmt **>(varlist_end()));
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_from;
}
};

class OMPUseDevicePtrClause final
: public OMPMappableExprListClause<OMPUseDevicePtrClause>,
private llvm::TrailingObjects<
OMPUseDevicePtrClause, Expr *, ValueDecl *, unsigned,
OMPClauseMappableExprCommon::MappableComponent> {
friend class OMPClauseReader;
friend OMPMappableExprListClause;
friend OMPVarListClause;
friend TrailingObjects;

explicit OMPUseDevicePtrClause(SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc, unsigned NumVars,
unsigned NumUniqueDeclarations,
unsigned NumComponentLists,
unsigned NumComponents)
: OMPMappableExprListClause(OMPC_use_device_ptr, StartLoc, LParenLoc,
EndLoc, NumVars, NumUniqueDeclarations,
NumComponentLists, NumComponents) {}

explicit OMPUseDevicePtrClause(unsigned NumVars,
unsigned NumUniqueDeclarations,
unsigned NumComponentLists,
unsigned NumComponents)
: OMPMappableExprListClause(OMPC_use_device_ptr, SourceLocation(),
SourceLocation(), SourceLocation(), NumVars,
NumUniqueDeclarations, NumComponentLists,
NumComponents) {}

size_t numTrailingObjects(OverloadToken<Expr *>) const {
return 3 * varlist_size();
}
size_t numTrailingObjects(OverloadToken<ValueDecl *>) const {
return getUniqueDeclarationsNum();
}
size_t numTrailingObjects(OverloadToken<unsigned>) const {
return getUniqueDeclarationsNum() + getTotalComponentListNum();
}

void setPrivateCopies(ArrayRef<Expr *> VL);

MutableArrayRef<Expr *> getPrivateCopies() {
return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
}
ArrayRef<const Expr *> getPrivateCopies() const {
return llvm::makeArrayRef(varlist_end(), varlist_size());
}

void setInits(ArrayRef<Expr *> VL);

MutableArrayRef<Expr *> getInits() {
return MutableArrayRef<Expr *>(getPrivateCopies().end(), varlist_size());
}
ArrayRef<const Expr *> getInits() const {
return llvm::makeArrayRef(getPrivateCopies().end(), varlist_size());
}

public:
static OMPUseDevicePtrClause *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc, ArrayRef<Expr *> Vars,
ArrayRef<Expr *> PrivateVars, ArrayRef<Expr *> Inits,
ArrayRef<ValueDecl *> Declarations,
MappableExprComponentListsRef ComponentLists);

static OMPUseDevicePtrClause *CreateEmpty(const ASTContext &C,
unsigned NumVars,
unsigned NumUniqueDeclarations,
unsigned NumComponentLists,
unsigned NumComponents);

using private_copies_iterator = MutableArrayRef<Expr *>::iterator;
using private_copies_const_iterator = ArrayRef<const Expr *>::iterator;
using private_copies_range = llvm::iterator_range<private_copies_iterator>;
using private_copies_const_range =
llvm::iterator_range<private_copies_const_iterator>;

private_copies_range private_copies() {
return private_copies_range(getPrivateCopies().begin(),
getPrivateCopies().end());
}

private_copies_const_range private_copies() const {
return private_copies_const_range(getPrivateCopies().begin(),
getPrivateCopies().end());
}

using inits_iterator = MutableArrayRef<Expr *>::iterator;
using inits_const_iterator = ArrayRef<const Expr *>::iterator;
using inits_range = llvm::iterator_range<inits_iterator>;
using inits_const_range = llvm::iterator_range<inits_const_iterator>;

inits_range inits() {
return inits_range(getInits().begin(), getInits().end());
}

inits_const_range inits() const {
return inits_const_range(getInits().begin(), getInits().end());
}

child_range children() {
return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
reinterpret_cast<Stmt **>(varlist_end()));
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_use_device_ptr;
}
};

class OMPIsDevicePtrClause final
: public OMPMappableExprListClause<OMPIsDevicePtrClause>,
private llvm::TrailingObjects<
OMPIsDevicePtrClause, Expr *, ValueDecl *, unsigned,
OMPClauseMappableExprCommon::MappableComponent> {
friend class OMPClauseReader;
friend OMPMappableExprListClause;
friend OMPVarListClause;
friend TrailingObjects;

explicit OMPIsDevicePtrClause(SourceLocation StartLoc,
SourceLocation LParenLoc, SourceLocation EndLoc,
unsigned NumVars,
unsigned NumUniqueDeclarations,
unsigned NumComponentLists,
unsigned NumComponents)
: OMPMappableExprListClause(OMPC_is_device_ptr, StartLoc, LParenLoc,
EndLoc, NumVars, NumUniqueDeclarations,
NumComponentLists, NumComponents) {}

explicit OMPIsDevicePtrClause(unsigned NumVars,
unsigned NumUniqueDeclarations,
unsigned NumComponentLists,
unsigned NumComponents)
: OMPMappableExprListClause(OMPC_is_device_ptr, SourceLocation(),
SourceLocation(), SourceLocation(), NumVars,
NumUniqueDeclarations, NumComponentLists,
NumComponents) {}

size_t numTrailingObjects(OverloadToken<Expr *>) const {
return varlist_size();
}
size_t numTrailingObjects(OverloadToken<ValueDecl *>) const {
return getUniqueDeclarationsNum();
}
size_t numTrailingObjects(OverloadToken<unsigned>) const {
return getUniqueDeclarationsNum() + getTotalComponentListNum();
}

public:
static OMPIsDevicePtrClause *
Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc, ArrayRef<Expr *> Vars,
ArrayRef<ValueDecl *> Declarations,
MappableExprComponentListsRef ComponentLists);

static OMPIsDevicePtrClause *CreateEmpty(const ASTContext &C,
unsigned NumVars,
unsigned NumUniqueDeclarations,
unsigned NumComponentLists,
unsigned NumComponents);

child_range children() {
return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
reinterpret_cast<Stmt **>(varlist_end()));
}

static bool classof(const OMPClause *T) {
return T->getClauseKind() == OMPC_is_device_ptr;
}
};

} 

#endif 
