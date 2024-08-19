
#ifndef LLVM_TRANSFORMS_VECTORIZE_LOOPVECTORIZATIONLEGALITY_H
#define LLVM_TRANSFORMS_VECTORIZE_LOOPVECTORIZATIONLEGALITY_H

#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

namespace llvm {

OptimizationRemarkAnalysis createLVMissedAnalysis(const char *PassName,
StringRef RemarkName,
Loop *TheLoop,
Instruction *I = nullptr);

class LoopVectorizeHints {
enum HintKind { HK_WIDTH, HK_UNROLL, HK_FORCE, HK_ISVECTORIZED };

struct Hint {
const char *Name;
unsigned Value; 
HintKind Kind;

Hint(const char *Name, unsigned Value, HintKind Kind)
: Name(Name), Value(Value), Kind(Kind) {}

bool validate(unsigned Val);
};

Hint Width;

Hint Interleave;

Hint Force;

Hint IsVectorized;

static StringRef Prefix() { return "llvm.loop."; }

bool PotentiallyUnsafe = false;

public:
enum ForceKind {
FK_Undefined = -1, 
FK_Disabled = 0,   
FK_Enabled = 1,    
};

LoopVectorizeHints(const Loop *L, bool DisableInterleaving,
OptimizationRemarkEmitter &ORE);

void setAlreadyVectorized() {
IsVectorized.Value = 1;
Hint Hints[] = {IsVectorized};
writeHintsToMetadata(Hints);
}

bool allowVectorization(Function *F, Loop *L, bool AlwaysVectorize) const;

void emitRemarkWithHints() const;

unsigned getWidth() const { return Width.Value; }
unsigned getInterleave() const { return Interleave.Value; }
unsigned getIsVectorized() const { return IsVectorized.Value; }
enum ForceKind getForce() const { return (ForceKind)Force.Value; }

const char *vectorizeAnalysisPassName() const;

bool allowReordering() const {
return getForce() == LoopVectorizeHints::FK_Enabled || getWidth() > 1;
}

bool isPotentiallyUnsafe() const {
return getForce() != LoopVectorizeHints::FK_Enabled && PotentiallyUnsafe;
}

void setPotentiallyUnsafe() { PotentiallyUnsafe = true; }

private:
void getHintsFromMetadata();

void setHint(StringRef Name, Metadata *Arg);

MDNode *createHintMetadata(StringRef Name, unsigned V) const;

bool matchesHintMetadataName(MDNode *Node, ArrayRef<Hint> HintTypes);

void writeHintsToMetadata(ArrayRef<Hint> HintTypes);

const Loop *TheLoop;

OptimizationRemarkEmitter &ORE;
};

class LoopVectorizationRequirements {
public:
LoopVectorizationRequirements(OptimizationRemarkEmitter &ORE) : ORE(ORE) {}

void addUnsafeAlgebraInst(Instruction *I) {
if (!UnsafeAlgebraInst)
UnsafeAlgebraInst = I;
}

void addRuntimePointerChecks(unsigned Num) { NumRuntimePointerChecks = Num; }

bool doesNotMeet(Function *F, Loop *L, const LoopVectorizeHints &Hints);

private:
unsigned NumRuntimePointerChecks = 0;
Instruction *UnsafeAlgebraInst = nullptr;

OptimizationRemarkEmitter &ORE;
};

class LoopVectorizationLegality {
public:
LoopVectorizationLegality(
Loop *L, PredicatedScalarEvolution &PSE, DominatorTree *DT,
TargetLibraryInfo *TLI, AliasAnalysis *AA, Function *F,
std::function<const LoopAccessInfo &(Loop &)> *GetLAA, LoopInfo *LI,
OptimizationRemarkEmitter *ORE, LoopVectorizationRequirements *R,
LoopVectorizeHints *H, DemandedBits *DB, AssumptionCache *AC)
: TheLoop(L), LI(LI), PSE(PSE), TLI(TLI), DT(DT), GetLAA(GetLAA),
ORE(ORE), Requirements(R), Hints(H), DB(DB), AC(AC) {}

using ReductionList = DenseMap<PHINode *, RecurrenceDescriptor>;

using InductionList = MapVector<PHINode *, InductionDescriptor>;

using RecurrenceSet = SmallPtrSet<const PHINode *, 8>;

bool canVectorize(bool UseVPlanNativePath);

PHINode *getPrimaryInduction() { return PrimaryInduction; }

ReductionList *getReductionVars() { return &Reductions; }

InductionList *getInductionVars() { return &Inductions; }

RecurrenceSet *getFirstOrderRecurrences() { return &FirstOrderRecurrences; }

DenseMap<Instruction *, Instruction *> &getSinkAfter() { return SinkAfter; }

Type *getWidestInductionType() { return WidestIndTy; }

bool isInductionPhi(const Value *V);

bool isCastedInductionVariable(const Value *V);

bool isInductionVariable(const Value *V);

bool isReductionVariable(PHINode *PN) { return Reductions.count(PN); }

bool isFirstOrderRecurrence(const PHINode *Phi);

bool blockNeedsPredication(BasicBlock *BB);

int isConsecutivePtr(Value *Ptr);

bool isUniform(Value *V);

const RuntimePointerChecking *getRuntimePointerChecking() const {
return LAI->getRuntimePointerChecking();
}

const LoopAccessInfo *getLAI() const { return LAI; }

unsigned getMaxSafeDepDistBytes() { return LAI->getMaxSafeDepDistBytes(); }

uint64_t getMaxSafeRegisterWidth() const {
return LAI->getDepChecker().getMaxSafeRegisterWidth();
}

bool hasStride(Value *V) { return LAI->hasStride(V); }

bool isMaskRequired(const Instruction *I) { return (MaskedOp.count(I) != 0); }

unsigned getNumStores() const { return LAI->getNumStores(); }
unsigned getNumLoads() const { return LAI->getNumLoads(); }

bool hasFunNoNaNAttr() const { return HasFunNoNaNAttr; }

private:
bool canVectorizeLoopNestCFG(Loop *Lp, bool UseVPlanNativePath);

bool canVectorizeLoopCFG(Loop *Lp, bool UseVPlanNativePath);

bool canVectorizeInstrs();

bool canVectorizeMemory();

bool canVectorizeWithIfConvert();

bool canVectorizeOuterLoop();

bool blockCanBePredicated(BasicBlock *BB, SmallPtrSetImpl<Value *> &SafePtrs);

void addInductionPhi(PHINode *Phi, const InductionDescriptor &ID,
SmallPtrSetImpl<Value *> &AllowedExit);

OptimizationRemarkAnalysis
createMissedAnalysis(StringRef RemarkName, Instruction *I = nullptr) const {
return createLVMissedAnalysis(Hints->vectorizeAnalysisPassName(),
RemarkName, TheLoop, I);
}

const ValueToValueMap *getSymbolicStrides() {
return LAI ? &LAI->getSymbolicStrides() : nullptr;
}

Loop *TheLoop;

LoopInfo *LI;

PredicatedScalarEvolution &PSE;

TargetLibraryInfo *TLI;

DominatorTree *DT;

std::function<const LoopAccessInfo &(Loop &)> *GetLAA;

const LoopAccessInfo *LAI = nullptr;

OptimizationRemarkEmitter *ORE;


PHINode *PrimaryInduction = nullptr;

ReductionList Reductions;

InductionList Inductions;

SmallPtrSet<Instruction *, 4> InductionCastsToIgnore;

RecurrenceSet FirstOrderRecurrences;

DenseMap<Instruction *, Instruction *> SinkAfter;

Type *WidestIndTy = nullptr;

SmallPtrSet<Value *, 4> AllowedExit;

bool HasFunNoNaNAttr = false;

LoopVectorizationRequirements *Requirements;

LoopVectorizeHints *Hints;

DemandedBits *DB;

AssumptionCache *AC;

SmallPtrSet<const Instruction *, 8> MaskedOp;
};

} 

#endif 
