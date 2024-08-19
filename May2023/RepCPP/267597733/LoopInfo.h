
#ifndef LLVM_ANALYSIS_LOOPINFO_H
#define LLVM_ANALYSIS_LOOPINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Allocator.h"
#include <algorithm>
#include <utility>

namespace llvm {

class DominatorTree;
class LoopInfo;
class Loop;
class MDNode;
class PHINode;
class raw_ostream;
template <class N, bool IsPostDom> class DominatorTreeBase;
template <class N, class M> class LoopInfoBase;
template <class N, class M> class LoopBase;

template <class BlockT, class LoopT> class LoopBase {
LoopT *ParentLoop;
std::vector<LoopT *> SubLoops;

std::vector<BlockT *> Blocks;

SmallPtrSet<const BlockT *, 8> DenseBlockSet;

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
bool IsInvalid = false;
#endif

LoopBase(const LoopBase<BlockT, LoopT> &) = delete;
const LoopBase<BlockT, LoopT> &
operator=(const LoopBase<BlockT, LoopT> &) = delete;

public:
unsigned getLoopDepth() const {
assert(!isInvalid() && "Loop not in a valid state!");
unsigned D = 1;
for (const LoopT *CurLoop = ParentLoop; CurLoop;
CurLoop = CurLoop->ParentLoop)
++D;
return D;
}
BlockT *getHeader() const { return getBlocks().front(); }
LoopT *getParentLoop() const { return ParentLoop; }

void setParentLoop(LoopT *L) {
assert(!isInvalid() && "Loop not in a valid state!");
ParentLoop = L;
}

bool contains(const LoopT *L) const {
assert(!isInvalid() && "Loop not in a valid state!");
if (L == this)
return true;
if (!L)
return false;
return contains(L->getParentLoop());
}

bool contains(const BlockT *BB) const {
assert(!isInvalid() && "Loop not in a valid state!");
return DenseBlockSet.count(BB);
}

template <class InstT> bool contains(const InstT *Inst) const {
return contains(Inst->getParent());
}

const std::vector<LoopT *> &getSubLoops() const {
assert(!isInvalid() && "Loop not in a valid state!");
return SubLoops;
}
std::vector<LoopT *> &getSubLoopsVector() {
assert(!isInvalid() && "Loop not in a valid state!");
return SubLoops;
}
typedef typename std::vector<LoopT *>::const_iterator iterator;
typedef
typename std::vector<LoopT *>::const_reverse_iterator reverse_iterator;
iterator begin() const { return getSubLoops().begin(); }
iterator end() const { return getSubLoops().end(); }
reverse_iterator rbegin() const { return getSubLoops().rbegin(); }
reverse_iterator rend() const { return getSubLoops().rend(); }
bool empty() const { return getSubLoops().empty(); }

ArrayRef<BlockT *> getBlocks() const {
assert(!isInvalid() && "Loop not in a valid state!");
return Blocks;
}
typedef typename ArrayRef<BlockT *>::const_iterator block_iterator;
block_iterator block_begin() const { return getBlocks().begin(); }
block_iterator block_end() const { return getBlocks().end(); }
inline iterator_range<block_iterator> blocks() const {
assert(!isInvalid() && "Loop not in a valid state!");
return make_range(block_begin(), block_end());
}

unsigned getNumBlocks() const {
assert(!isInvalid() && "Loop not in a valid state!");
return Blocks.size();
}

std::vector<BlockT *> &getBlocksVector() {
assert(!isInvalid() && "Loop not in a valid state!");
return Blocks;
}
SmallPtrSetImpl<const BlockT *> &getBlocksSet() {
assert(!isInvalid() && "Loop not in a valid state!");
return DenseBlockSet;
}

const SmallPtrSetImpl<const BlockT *> &getBlocksSet() const {
assert(!isInvalid() && "Loop not in a valid state!");
return DenseBlockSet;
}

bool isInvalid() const {
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
return IsInvalid;
#else
return false;
#endif
}

bool isLoopExiting(const BlockT *BB) const {
assert(!isInvalid() && "Loop not in a valid state!");
for (const auto &Succ : children<const BlockT *>(BB)) {
if (!contains(Succ))
return true;
}
return false;
}

bool isLoopLatch(const BlockT *BB) const {
assert(!isInvalid() && "Loop not in a valid state!");
assert(contains(BB) && "block does not belong to the loop");

BlockT *Header = getHeader();
auto PredBegin = GraphTraits<Inverse<BlockT *>>::child_begin(Header);
auto PredEnd = GraphTraits<Inverse<BlockT *>>::child_end(Header);
return std::find(PredBegin, PredEnd, BB) != PredEnd;
}

unsigned getNumBackEdges() const {
assert(!isInvalid() && "Loop not in a valid state!");
unsigned NumBackEdges = 0;
BlockT *H = getHeader();

for (const auto Pred : children<Inverse<BlockT *>>(H))
if (contains(Pred))
++NumBackEdges;

return NumBackEdges;
}


void getExitingBlocks(SmallVectorImpl<BlockT *> &ExitingBlocks) const;

BlockT *getExitingBlock() const;

void getExitBlocks(SmallVectorImpl<BlockT *> &ExitBlocks) const;

BlockT *getExitBlock() const;

bool hasDedicatedExits() const;

void getUniqueExitBlocks(SmallVectorImpl<BlockT *> &ExitBlocks) const;

BlockT *getUniqueExitBlock() const;

typedef std::pair<const BlockT *, const BlockT *> Edge;

void getExitEdges(SmallVectorImpl<Edge> &ExitEdges) const;

BlockT *getLoopPreheader() const;

BlockT *getLoopPredecessor() const;

BlockT *getLoopLatch() const;

void getLoopLatches(SmallVectorImpl<BlockT *> &LoopLatches) const {
assert(!isInvalid() && "Loop not in a valid state!");
BlockT *H = getHeader();
for (const auto Pred : children<Inverse<BlockT *>>(H))
if (contains(Pred))
LoopLatches.push_back(Pred);
}


void addBasicBlockToLoop(BlockT *NewBB, LoopInfoBase<BlockT, LoopT> &LI);

void replaceChildLoopWith(LoopT *OldChild, LoopT *NewChild);

void addChildLoop(LoopT *NewChild) {
assert(!isInvalid() && "Loop not in a valid state!");
assert(!NewChild->ParentLoop && "NewChild already has a parent!");
NewChild->ParentLoop = static_cast<LoopT *>(this);
SubLoops.push_back(NewChild);
}

LoopT *removeChildLoop(iterator I) {
assert(!isInvalid() && "Loop not in a valid state!");
assert(I != SubLoops.end() && "Cannot remove end iterator!");
LoopT *Child = *I;
assert(Child->ParentLoop == this && "Child is not a child of this loop!");
SubLoops.erase(SubLoops.begin() + (I - begin()));
Child->ParentLoop = nullptr;
return Child;
}

LoopT *removeChildLoop(LoopT *Child) {
return removeChildLoop(llvm::find(*this, Child));
}

void addBlockEntry(BlockT *BB) {
assert(!isInvalid() && "Loop not in a valid state!");
Blocks.push_back(BB);
DenseBlockSet.insert(BB);
}

void reverseBlock(unsigned from) {
assert(!isInvalid() && "Loop not in a valid state!");
std::reverse(Blocks.begin() + from, Blocks.end());
}

void reserveBlocks(unsigned size) {
assert(!isInvalid() && "Loop not in a valid state!");
Blocks.reserve(size);
}

void moveToHeader(BlockT *BB) {
assert(!isInvalid() && "Loop not in a valid state!");
if (Blocks[0] == BB)
return;
for (unsigned i = 0;; ++i) {
assert(i != Blocks.size() && "Loop does not contain BB!");
if (Blocks[i] == BB) {
Blocks[i] = Blocks[0];
Blocks[0] = BB;
return;
}
}
}

void removeBlockFromLoop(BlockT *BB) {
assert(!isInvalid() && "Loop not in a valid state!");
auto I = find(Blocks, BB);
assert(I != Blocks.end() && "N is not in this list!");
Blocks.erase(I);

DenseBlockSet.erase(BB);
}

void verifyLoop() const;

void verifyLoopNest(DenseSet<const LoopT *> *Loops) const;

void print(raw_ostream &OS, unsigned Depth = 0, bool Verbose = false) const;

protected:
friend class LoopInfoBase<BlockT, LoopT>;

LoopBase() : ParentLoop(nullptr) {}

explicit LoopBase(BlockT *BB) : ParentLoop(nullptr) {
Blocks.push_back(BB);
DenseBlockSet.insert(BB);
}

~LoopBase() {
for (auto *SubLoop : SubLoops)
SubLoop->~LoopT();

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
IsInvalid = true;
#endif
SubLoops.clear();
Blocks.clear();
DenseBlockSet.clear();
ParentLoop = nullptr;
}
};

template <class BlockT, class LoopT>
raw_ostream &operator<<(raw_ostream &OS, const LoopBase<BlockT, LoopT> &Loop) {
Loop.print(OS);
return OS;
}

extern template class LoopBase<BasicBlock, Loop>;

class Loop : public LoopBase<BasicBlock, Loop> {
public:
class LocRange {
DebugLoc Start;
DebugLoc End;

public:
LocRange() {}
LocRange(DebugLoc Start) : Start(std::move(Start)), End(std::move(Start)) {}
LocRange(DebugLoc Start, DebugLoc End)
: Start(std::move(Start)), End(std::move(End)) {}

const DebugLoc &getStart() const { return Start; }
const DebugLoc &getEnd() const { return End; }

explicit operator bool() const { return Start && End; }
};

bool isLoopInvariant(const Value *V) const;

bool hasLoopInvariantOperands(const Instruction *I) const;

bool makeLoopInvariant(Value *V, bool &Changed,
Instruction *InsertPt = nullptr) const;

bool makeLoopInvariant(Instruction *I, bool &Changed,
Instruction *InsertPt = nullptr) const;

PHINode *getCanonicalInductionVariable() const;

bool isLCSSAForm(DominatorTree &DT) const;

bool isRecursivelyLCSSAForm(DominatorTree &DT, const LoopInfo &LI) const;

bool isLoopSimplifyForm() const;

bool isSafeToClone() const;

bool isAnnotatedParallel() const;

MDNode *getLoopID() const;
void setLoopID(MDNode *LoopID) const;

void setLoopAlreadyUnrolled();

void dump() const;
void dumpVerbose() const;

DebugLoc getStartLoc() const;

LocRange getLocRange() const;

StringRef getName() const {
if (BasicBlock *Header = getHeader())
if (Header->hasName())
return Header->getName();
return "<unnamed loop>";
}

private:
Loop() = default;

friend class LoopInfoBase<BasicBlock, Loop>;
friend class LoopBase<BasicBlock, Loop>;
explicit Loop(BasicBlock *BB) : LoopBase<BasicBlock, Loop>(BB) {}
~Loop() = default;
};


template <class BlockT, class LoopT> class LoopInfoBase {
DenseMap<const BlockT *, LoopT *> BBMap;
std::vector<LoopT *> TopLevelLoops;
BumpPtrAllocator LoopAllocator;

friend class LoopBase<BlockT, LoopT>;
friend class LoopInfo;

void operator=(const LoopInfoBase &) = delete;
LoopInfoBase(const LoopInfoBase &) = delete;

public:
LoopInfoBase() {}
~LoopInfoBase() { releaseMemory(); }

LoopInfoBase(LoopInfoBase &&Arg)
: BBMap(std::move(Arg.BBMap)),
TopLevelLoops(std::move(Arg.TopLevelLoops)),
LoopAllocator(std::move(Arg.LoopAllocator)) {
Arg.TopLevelLoops.clear();
}
LoopInfoBase &operator=(LoopInfoBase &&RHS) {
BBMap = std::move(RHS.BBMap);

for (auto *L : TopLevelLoops)
L->~LoopT();

TopLevelLoops = std::move(RHS.TopLevelLoops);
LoopAllocator = std::move(RHS.LoopAllocator);
RHS.TopLevelLoops.clear();
return *this;
}

void releaseMemory() {
BBMap.clear();

for (auto *L : TopLevelLoops)
L->~LoopT();
TopLevelLoops.clear();
LoopAllocator.Reset();
}

template <typename... ArgsTy> LoopT *AllocateLoop(ArgsTy &&... Args) {
LoopT *Storage = LoopAllocator.Allocate<LoopT>();
return new (Storage) LoopT(std::forward<ArgsTy>(Args)...);
}

typedef typename std::vector<LoopT *>::const_iterator iterator;
typedef
typename std::vector<LoopT *>::const_reverse_iterator reverse_iterator;
iterator begin() const { return TopLevelLoops.begin(); }
iterator end() const { return TopLevelLoops.end(); }
reverse_iterator rbegin() const { return TopLevelLoops.rbegin(); }
reverse_iterator rend() const { return TopLevelLoops.rend(); }
bool empty() const { return TopLevelLoops.empty(); }

SmallVector<LoopT *, 4> getLoopsInPreorder();

SmallVector<LoopT *, 4> getLoopsInReverseSiblingPreorder();

LoopT *getLoopFor(const BlockT *BB) const { return BBMap.lookup(BB); }

const LoopT *operator[](const BlockT *BB) const { return getLoopFor(BB); }

unsigned getLoopDepth(const BlockT *BB) const {
const LoopT *L = getLoopFor(BB);
return L ? L->getLoopDepth() : 0;
}

bool isLoopHeader(const BlockT *BB) const {
const LoopT *L = getLoopFor(BB);
return L && L->getHeader() == BB;
}

LoopT *removeLoop(iterator I) {
assert(I != end() && "Cannot remove end iterator!");
LoopT *L = *I;
assert(!L->getParentLoop() && "Not a top-level loop!");
TopLevelLoops.erase(TopLevelLoops.begin() + (I - begin()));
return L;
}

void changeLoopFor(BlockT *BB, LoopT *L) {
if (!L) {
BBMap.erase(BB);
return;
}
BBMap[BB] = L;
}

void changeTopLevelLoop(LoopT *OldLoop, LoopT *NewLoop) {
auto I = find(TopLevelLoops, OldLoop);
assert(I != TopLevelLoops.end() && "Old loop not at top level!");
*I = NewLoop;
assert(!NewLoop->ParentLoop && !OldLoop->ParentLoop &&
"Loops already embedded into a subloop!");
}

void addTopLevelLoop(LoopT *New) {
assert(!New->getParentLoop() && "Loop already in subloop!");
TopLevelLoops.push_back(New);
}

void removeBlock(BlockT *BB) {
auto I = BBMap.find(BB);
if (I != BBMap.end()) {
for (LoopT *L = I->second; L; L = L->getParentLoop())
L->removeBlockFromLoop(BB);

BBMap.erase(I);
}
}


static bool isNotAlreadyContainedIn(const LoopT *SubLoop,
const LoopT *ParentLoop) {
if (!SubLoop)
return true;
if (SubLoop == ParentLoop)
return false;
return isNotAlreadyContainedIn(SubLoop->getParentLoop(), ParentLoop);
}

void analyze(const DominatorTreeBase<BlockT, false> &DomTree);

void print(raw_ostream &OS) const;

void verify(const DominatorTreeBase<BlockT, false> &DomTree) const;

void destroy(LoopT *L) {
L->~LoopT();

LoopAllocator.Deallocate(L);
}
};

extern template class LoopInfoBase<BasicBlock, Loop>;

class LoopInfo : public LoopInfoBase<BasicBlock, Loop> {
typedef LoopInfoBase<BasicBlock, Loop> BaseT;

friend class LoopBase<BasicBlock, Loop>;

void operator=(const LoopInfo &) = delete;
LoopInfo(const LoopInfo &) = delete;

public:
LoopInfo() {}
explicit LoopInfo(const DominatorTreeBase<BasicBlock, false> &DomTree);

LoopInfo(LoopInfo &&Arg) : BaseT(std::move(static_cast<BaseT &>(Arg))) {}
LoopInfo &operator=(LoopInfo &&RHS) {
BaseT::operator=(std::move(static_cast<BaseT &>(RHS)));
return *this;
}

bool invalidate(Function &F, const PreservedAnalyses &PA,
FunctionAnalysisManager::Invalidator &);


void erase(Loop *L);

bool replacementPreservesLCSSAForm(Instruction *From, Value *To) {
Instruction *I = dyn_cast<Instruction>(To);
if (!I)
return true;
if (I->getParent() == From->getParent())
return true;
Loop *ToLoop = getLoopFor(I->getParent());
if (!ToLoop)
return true;
return ToLoop->contains(getLoopFor(From->getParent()));
}

bool movementPreservesLCSSAForm(Instruction *Inst, Instruction *NewLoc) {
assert(Inst->getFunction() == NewLoc->getFunction() &&
"Can't reason about IPO!");

auto *OldBB = Inst->getParent();
auto *NewBB = NewLoc->getParent();

if (OldBB == NewBB)
return true;

auto *OldLoop = getLoopFor(OldBB);
auto *NewLoop = getLoopFor(NewBB);

if (OldLoop == NewLoop)
return true;

auto Contains = [](const Loop *Outer, const Loop *Inner) {
return !Outer || Outer->contains(Inner);
};



if (!Contains(NewLoop, OldLoop)) {
for (Use &U : Inst->uses()) {
auto *UI = cast<Instruction>(U.getUser());
auto *UBB = isa<PHINode>(UI) ? cast<PHINode>(UI)->getIncomingBlock(U)
: UI->getParent();
if (UBB != NewBB && getLoopFor(UBB) != NewLoop)
return false;
}
}


if (!Contains(OldLoop, NewLoop)) {
if (isa<PHINode>(Inst))
return false;

for (Use &U : Inst->operands()) {
auto *DefI = dyn_cast<Instruction>(U.get());
if (!DefI)
return false;


auto *DefBlock = DefI->getParent();
if (DefBlock != NewBB && getLoopFor(DefBlock) != NewLoop)
return false;
}
}

return true;
}
};

template <> struct GraphTraits<const Loop *> {
typedef const Loop *NodeRef;
typedef LoopInfo::iterator ChildIteratorType;

static NodeRef getEntryNode(const Loop *L) { return L; }
static ChildIteratorType child_begin(NodeRef N) { return N->begin(); }
static ChildIteratorType child_end(NodeRef N) { return N->end(); }
};

template <> struct GraphTraits<Loop *> {
typedef Loop *NodeRef;
typedef LoopInfo::iterator ChildIteratorType;

static NodeRef getEntryNode(Loop *L) { return L; }
static ChildIteratorType child_begin(NodeRef N) { return N->begin(); }
static ChildIteratorType child_end(NodeRef N) { return N->end(); }
};

class LoopAnalysis : public AnalysisInfoMixin<LoopAnalysis> {
friend AnalysisInfoMixin<LoopAnalysis>;
static AnalysisKey Key;

public:
typedef LoopInfo Result;

LoopInfo run(Function &F, FunctionAnalysisManager &AM);
};

class LoopPrinterPass : public PassInfoMixin<LoopPrinterPass> {
raw_ostream &OS;

public:
explicit LoopPrinterPass(raw_ostream &OS) : OS(OS) {}
PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

struct LoopVerifierPass : public PassInfoMixin<LoopVerifierPass> {
PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

class LoopInfoWrapperPass : public FunctionPass {
LoopInfo LI;

public:
static char ID; 

LoopInfoWrapperPass() : FunctionPass(ID) {
initializeLoopInfoWrapperPassPass(*PassRegistry::getPassRegistry());
}

LoopInfo &getLoopInfo() { return LI; }
const LoopInfo &getLoopInfo() const { return LI; }

bool runOnFunction(Function &F) override;

void verifyAnalysis() const override;

void releaseMemory() override { LI.releaseMemory(); }

void print(raw_ostream &O, const Module *M = nullptr) const override;

void getAnalysisUsage(AnalysisUsage &AU) const override;
};

void printLoop(Loop &L, raw_ostream &OS, const std::string &Banner = "");

} 

#endif
