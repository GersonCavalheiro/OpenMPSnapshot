
#ifndef LLVM_ANALYSIS_MEMORYDEPENDENCEANALYSIS_H
#define LLVM_ANALYSIS_MEMORYDEPENDENCEANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/PointerSumType.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PredIteratorCache.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

namespace llvm {

class AssumptionCache;
class CallSite;
class DominatorTree;
class Function;
class Instruction;
class LoadInst;
class PHITransAddr;
class TargetLibraryInfo;
class PhiValues;
class Value;

class MemDepResult {
enum DepType {
Invalid = 0,

Clobber,

Def,

Other
};

enum OtherType {
NonLocal = 1,
NonFuncLocal,
Unknown
};

using ValueTy = PointerSumType<
DepType, PointerSumTypeMember<Invalid, Instruction *>,
PointerSumTypeMember<Clobber, Instruction *>,
PointerSumTypeMember<Def, Instruction *>,
PointerSumTypeMember<Other, PointerEmbeddedInt<OtherType, 3>>>;
ValueTy Value;

explicit MemDepResult(ValueTy V) : Value(V) {}

public:
MemDepResult() = default;

static MemDepResult getDef(Instruction *Inst) {
assert(Inst && "Def requires inst");
return MemDepResult(ValueTy::create<Def>(Inst));
}
static MemDepResult getClobber(Instruction *Inst) {
assert(Inst && "Clobber requires inst");
return MemDepResult(ValueTy::create<Clobber>(Inst));
}
static MemDepResult getNonLocal() {
return MemDepResult(ValueTy::create<Other>(NonLocal));
}
static MemDepResult getNonFuncLocal() {
return MemDepResult(ValueTy::create<Other>(NonFuncLocal));
}
static MemDepResult getUnknown() {
return MemDepResult(ValueTy::create<Other>(Unknown));
}

bool isClobber() const { return Value.is<Clobber>(); }

bool isDef() const { return Value.is<Def>(); }

bool isNonLocal() const {
return Value.is<Other>() && Value.cast<Other>() == NonLocal;
}

bool isNonFuncLocal() const {
return Value.is<Other>() && Value.cast<Other>() == NonFuncLocal;
}

bool isUnknown() const {
return Value.is<Other>() && Value.cast<Other>() == Unknown;
}

Instruction *getInst() const {
switch (Value.getTag()) {
case Invalid:
return Value.cast<Invalid>();
case Clobber:
return Value.cast<Clobber>();
case Def:
return Value.cast<Def>();
case Other:
return nullptr;
}
llvm_unreachable("Unknown discriminant!");
}

bool operator==(const MemDepResult &M) const { return Value == M.Value; }
bool operator!=(const MemDepResult &M) const { return Value != M.Value; }
bool operator<(const MemDepResult &M) const { return Value < M.Value; }
bool operator>(const MemDepResult &M) const { return Value > M.Value; }

private:
friend class MemoryDependenceResults;

bool isDirty() const { return Value.is<Invalid>(); }

static MemDepResult getDirty(Instruction *Inst) {
return MemDepResult(ValueTy::create<Invalid>(Inst));
}
};

class NonLocalDepEntry {
BasicBlock *BB;
MemDepResult Result;

public:
NonLocalDepEntry(BasicBlock *bb, MemDepResult result)
: BB(bb), Result(result) {}

NonLocalDepEntry(BasicBlock *bb) : BB(bb) {}

BasicBlock *getBB() const { return BB; }

void setResult(const MemDepResult &R) { Result = R; }

const MemDepResult &getResult() const { return Result; }

bool operator<(const NonLocalDepEntry &RHS) const { return BB < RHS.BB; }
};

class NonLocalDepResult {
NonLocalDepEntry Entry;
Value *Address;

public:
NonLocalDepResult(BasicBlock *bb, MemDepResult result, Value *address)
: Entry(bb, result), Address(address) {}

BasicBlock *getBB() const { return Entry.getBB(); }

void setResult(const MemDepResult &R, Value *Addr) {
Entry.setResult(R);
Address = Addr;
}

const MemDepResult &getResult() const { return Entry.getResult(); }

Value *getAddress() const { return Address; }
};

class MemoryDependenceResults {
using LocalDepMapType = DenseMap<Instruction *, MemDepResult>;
LocalDepMapType LocalDeps;

public:
using NonLocalDepInfo = std::vector<NonLocalDepEntry>;

private:
using ValueIsLoadPair = PointerIntPair<const Value *, 1, bool>;

using BBSkipFirstBlockPair = PointerIntPair<BasicBlock *, 1, bool>;

struct NonLocalPointerInfo {
BBSkipFirstBlockPair Pair;
NonLocalDepInfo NonLocalDeps;
LocationSize Size = MemoryLocation::UnknownSize;
AAMDNodes AATags;

NonLocalPointerInfo() = default;
};

DenseMap<AssertingVH<const Value>, NonLocalDepResult> NonLocalDefsCache;
using ReverseNonLocalDefsCacheTy =
DenseMap<Instruction *, SmallPtrSet<const Value*, 4>>;
ReverseNonLocalDefsCacheTy ReverseNonLocalDefsCache;

using CachedNonLocalPointerInfo =
DenseMap<ValueIsLoadPair, NonLocalPointerInfo>;
CachedNonLocalPointerInfo NonLocalPointerDeps;

using ReverseNonLocalPtrDepTy =
DenseMap<Instruction *, SmallPtrSet<ValueIsLoadPair, 4>>;
ReverseNonLocalPtrDepTy ReverseNonLocalPtrDeps;

using PerInstNLInfo = std::pair<NonLocalDepInfo, bool>;

using NonLocalDepMapType = DenseMap<Instruction *, PerInstNLInfo>;

NonLocalDepMapType NonLocalDeps;

using ReverseDepMapType =
DenseMap<Instruction *, SmallPtrSet<Instruction *, 4>>;
ReverseDepMapType ReverseLocalDeps;

ReverseDepMapType ReverseNonLocalDeps;

AliasAnalysis &AA;
AssumptionCache &AC;
const TargetLibraryInfo &TLI;
DominatorTree &DT;
PhiValues &PV;
PredIteratorCache PredCache;

public:
MemoryDependenceResults(AliasAnalysis &AA, AssumptionCache &AC,
const TargetLibraryInfo &TLI,
DominatorTree &DT, PhiValues &PV)
: AA(AA), AC(AC), TLI(TLI), DT(DT), PV(PV) {}

bool invalidate(Function &F, const PreservedAnalyses &PA,
FunctionAnalysisManager::Invalidator &Inv);

unsigned getDefaultBlockScanLimit() const;

MemDepResult getDependency(Instruction *QueryInst);

const NonLocalDepInfo &getNonLocalCallDependency(CallSite QueryCS);

void getNonLocalPointerDependency(Instruction *QueryInst,
SmallVectorImpl<NonLocalDepResult> &Result);

void removeInstruction(Instruction *InstToRemove);

void invalidateCachedPointerInfo(Value *Ptr);

void invalidateCachedPredecessors();

MemDepResult getPointerDependencyFrom(const MemoryLocation &Loc, bool isLoad,
BasicBlock::iterator ScanIt,
BasicBlock *BB,
Instruction *QueryInst = nullptr,
unsigned *Limit = nullptr);

MemDepResult getSimplePointerDependencyFrom(const MemoryLocation &MemLoc,
bool isLoad,
BasicBlock::iterator ScanIt,
BasicBlock *BB,
Instruction *QueryInst,
unsigned *Limit = nullptr);

MemDepResult getInvariantGroupPointerDependency(LoadInst *LI, BasicBlock *BB);

static unsigned getLoadLoadClobberFullWidthSize(const Value *MemLocBase,
int64_t MemLocOffs,
unsigned MemLocSize,
const LoadInst *LI);

void releaseMemory();

private:
MemDepResult getCallSiteDependencyFrom(CallSite C, bool isReadOnlyCall,
BasicBlock::iterator ScanIt,
BasicBlock *BB);
bool getNonLocalPointerDepFromBB(Instruction *QueryInst,
const PHITransAddr &Pointer,
const MemoryLocation &Loc, bool isLoad,
BasicBlock *BB,
SmallVectorImpl<NonLocalDepResult> &Result,
DenseMap<BasicBlock *, Value *> &Visited,
bool SkipFirstBlock = false);
MemDepResult GetNonLocalInfoForBlock(Instruction *QueryInst,
const MemoryLocation &Loc, bool isLoad,
BasicBlock *BB, NonLocalDepInfo *Cache,
unsigned NumSortedEntries);

void RemoveCachedNonLocalPointerDependencies(ValueIsLoadPair P);

void verifyRemoved(Instruction *Inst) const;
};

class MemoryDependenceAnalysis
: public AnalysisInfoMixin<MemoryDependenceAnalysis> {
friend AnalysisInfoMixin<MemoryDependenceAnalysis>;

static AnalysisKey Key;

public:
using Result = MemoryDependenceResults;

MemoryDependenceResults run(Function &F, FunctionAnalysisManager &AM);
};

class MemoryDependenceWrapperPass : public FunctionPass {
Optional<MemoryDependenceResults> MemDep;

public:
static char ID;

MemoryDependenceWrapperPass();
~MemoryDependenceWrapperPass() override;

bool runOnFunction(Function &) override;

void releaseMemory() override;

void getAnalysisUsage(AnalysisUsage &AU) const override;

MemoryDependenceResults &getMemDep() { return *MemDep; }
};

} 

#endif 
