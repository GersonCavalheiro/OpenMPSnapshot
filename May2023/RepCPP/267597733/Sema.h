
#ifndef LLVM_CLANG_SEMA_SEMA_H
#define LLVM_CLANG_SEMA_SEMA_H

#include "clang/AST/Attr.h"
#include "clang/AST/Availability.h"
#include "clang/AST/ComparisonCategories.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/LocInfoType.h"
#include "clang/AST/MangleNumberingContext.h"
#include "clang/AST/NSAPI.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/TypeLoc.h"
#include "clang/AST/TypeOrdering.h"
#include "clang/Basic/ExpressionTraits.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/OpenMPKinds.h"
#include "clang/Basic/PragmaKinds.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TemplateKinds.h"
#include "clang/Basic/TypeTraits.h"
#include "clang/Sema/AnalysisBasedWarnings.h"
#include "clang/Sema/CleanupInfo.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/IdentifierResolver.h"
#include "clang/Sema/ObjCMethodList.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/TypoCorrection.h"
#include "clang/Sema/Weak.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TinyPtrVector.h"
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
class APSInt;
template <typename ValueT> struct DenseMapInfo;
template <typename ValueT, typename ValueInfoT> class DenseSet;
class SmallBitVector;
struct InlineAsmIdentifierInfo;
}

namespace clang {
class ADLResult;
class ASTConsumer;
class ASTContext;
class ASTMutationListener;
class ASTReader;
class ASTWriter;
class ArrayType;
class ParsedAttr;
class BindingDecl;
class BlockDecl;
class CapturedDecl;
class CXXBasePath;
class CXXBasePaths;
class CXXBindTemporaryExpr;
typedef SmallVector<CXXBaseSpecifier*, 4> CXXCastPath;
class CXXConstructorDecl;
class CXXConversionDecl;
class CXXDeleteExpr;
class CXXDestructorDecl;
class CXXFieldCollector;
class CXXMemberCallExpr;
class CXXMethodDecl;
class CXXScopeSpec;
class CXXTemporary;
class CXXTryStmt;
class CallExpr;
class ClassTemplateDecl;
class ClassTemplatePartialSpecializationDecl;
class ClassTemplateSpecializationDecl;
class VarTemplatePartialSpecializationDecl;
class CodeCompleteConsumer;
class CodeCompletionAllocator;
class CodeCompletionTUInfo;
class CodeCompletionResult;
class CoroutineBodyStmt;
class Decl;
class DeclAccessPair;
class DeclContext;
class DeclRefExpr;
class DeclaratorDecl;
class DeducedTemplateArgument;
class DependentDiagnostic;
class DesignatedInitExpr;
class Designation;
class EnableIfAttr;
class EnumConstantDecl;
class Expr;
class ExtVectorType;
class FormatAttr;
class FriendDecl;
class FunctionDecl;
class FunctionProtoType;
class FunctionTemplateDecl;
class ImplicitConversionSequence;
typedef MutableArrayRef<ImplicitConversionSequence> ConversionSequenceList;
class InitListExpr;
class InitializationKind;
class InitializationSequence;
class InitializedEntity;
class IntegerLiteral;
class LabelStmt;
class LambdaExpr;
class LangOptions;
class LocalInstantiationScope;
class LookupResult;
class MacroInfo;
typedef ArrayRef<std::pair<IdentifierInfo *, SourceLocation>> ModuleIdPath;
class ModuleLoader;
class MultiLevelTemplateArgumentList;
class NamedDecl;
class ObjCCategoryDecl;
class ObjCCategoryImplDecl;
class ObjCCompatibleAliasDecl;
class ObjCContainerDecl;
class ObjCImplDecl;
class ObjCImplementationDecl;
class ObjCInterfaceDecl;
class ObjCIvarDecl;
template <class T> class ObjCList;
class ObjCMessageExpr;
class ObjCMethodDecl;
class ObjCPropertyDecl;
class ObjCProtocolDecl;
class OMPThreadPrivateDecl;
class OMPDeclareReductionDecl;
class OMPDeclareSimdDecl;
class OMPClause;
struct OverloadCandidate;
class OverloadCandidateSet;
class OverloadExpr;
class ParenListExpr;
class ParmVarDecl;
class Preprocessor;
class PseudoDestructorTypeStorage;
class PseudoObjectExpr;
class QualType;
class StandardConversionSequence;
class Stmt;
class StringLiteral;
class SwitchStmt;
class TemplateArgument;
class TemplateArgumentList;
class TemplateArgumentLoc;
class TemplateDecl;
class TemplateInstantiationCallback;
class TemplateParameterList;
class TemplatePartialOrderingContext;
class TemplateTemplateParmDecl;
class Token;
class TypeAliasDecl;
class TypedefDecl;
class TypedefNameDecl;
class TypeLoc;
class TypoCorrectionConsumer;
class UnqualifiedId;
class UnresolvedLookupExpr;
class UnresolvedMemberExpr;
class UnresolvedSetImpl;
class UnresolvedSetIterator;
class UsingDecl;
class UsingShadowDecl;
class ValueDecl;
class VarDecl;
class VarTemplateSpecializationDecl;
class VisibilityAttr;
class VisibleDeclConsumer;
class IndirectFieldDecl;
struct DeductionFailureInfo;
class TemplateSpecCandidateSet;

namespace sema {
class AccessedEntity;
class BlockScopeInfo;
class Capture;
class CapturedRegionScopeInfo;
class CapturingScopeInfo;
class CompoundScopeInfo;
class DelayedDiagnostic;
class DelayedDiagnosticPool;
class FunctionScopeInfo;
class LambdaScopeInfo;
class PossiblyUnreachableDiag;
class SemaPPCallbacks;
class TemplateDeductionInfo;
}

namespace threadSafety {
class BeforeSet;
void threadSafetyCleanup(BeforeSet* Cache);
}

typedef std::pair<llvm::PointerUnion<const TemplateTypeParmType*, NamedDecl*>,
SourceLocation> UnexpandedParameterPack;

struct FileNullability {
SourceLocation PointerLoc;

SourceLocation PointerEndLoc;

uint8_t PointerKind;

bool SawTypeNullability = false;
};

class FileNullabilityMap {
llvm::DenseMap<FileID, FileNullability> Map;

struct {
FileID File;
FileNullability Nullability;
} Cache;

public:
FileNullability &operator[](FileID file) {
if (file == Cache.File)
return Cache.Nullability;

if (!Cache.File.isInvalid()) {
Map[Cache.File] = Cache.Nullability;
}

Cache.File = file;
Cache.Nullability = Map[file];
return Cache.Nullability;
}
};

class Sema {
Sema(const Sema &) = delete;
void operator=(const Sema &) = delete;

ExternalSemaSource *ExternalSource;

bool isMultiplexExternalSource;

static bool mightHaveNonExternalLinkage(const DeclaratorDecl *FD);

bool isVisibleSlow(const NamedDecl *D);

bool shouldLinkPossiblyHiddenDecl(const NamedDecl *Old,
const NamedDecl *New) {
if (isVisible(Old))
return true;
if (New->isExternallyDeclarable()) {
assert(Old->isExternallyDeclarable() &&
"should not have found a non-externally-declarable previous decl");
return true;
}
return false;
}
bool shouldLinkPossiblyHiddenDecl(LookupResult &Old, const NamedDecl *New);

public:
typedef OpaquePtr<DeclGroupRef> DeclGroupPtrTy;
typedef OpaquePtr<TemplateName> TemplateTy;
typedef OpaquePtr<QualType> TypeTy;

OpenCLOptions OpenCLFeatures;
FPOptions FPFeatures;

const LangOptions &LangOpts;
Preprocessor &PP;
ASTContext &Context;
ASTConsumer &Consumer;
DiagnosticsEngine &Diags;
SourceManager &SourceMgr;

bool CollectStats;

CodeCompleteConsumer *CodeCompleter;

DeclContext *CurContext;

DeclContext *OriginalLexicalContext;

DeclarationName VAListTagName;

bool MSStructPragmaOn; 

LangOptions::PragmaMSPointersToMembersKind
MSPointerToMemberRepresentationMethod;

SmallVector<Scope*, 2> CurrentSEHFinally;

SourceLocation ImplicitMSInheritanceAttrLoc;

enum PragmaClangSectionKind {
PCSK_Invalid      = 0,
PCSK_BSS          = 1,
PCSK_Data         = 2,
PCSK_Rodata       = 3,
PCSK_Text         = 4
};

enum PragmaClangSectionAction {
PCSA_Set     = 0,
PCSA_Clear   = 1
};

struct PragmaClangSection {
std::string SectionName;
bool Valid = false;
SourceLocation PragmaLocation;

void Act(SourceLocation PragmaLocation,
PragmaClangSectionAction Action,
StringLiteral* Name);
};

PragmaClangSection PragmaClangBSSSection;
PragmaClangSection PragmaClangDataSection;
PragmaClangSection PragmaClangRodataSection;
PragmaClangSection PragmaClangTextSection;

enum PragmaMsStackAction {
PSK_Reset     = 0x0,                
PSK_Set       = 0x1,                
PSK_Push      = 0x2,                
PSK_Pop       = 0x4,                
PSK_Show      = 0x8,                
PSK_Push_Set  = PSK_Push | PSK_Set, 
PSK_Pop_Set   = PSK_Pop | PSK_Set,  
};

template<typename ValueType>
struct PragmaStack {
struct Slot {
llvm::StringRef StackSlotLabel;
ValueType Value;
SourceLocation PragmaLocation;
SourceLocation PragmaPushLocation;
Slot(llvm::StringRef StackSlotLabel, ValueType Value,
SourceLocation PragmaLocation, SourceLocation PragmaPushLocation)
: StackSlotLabel(StackSlotLabel), Value(Value),
PragmaLocation(PragmaLocation),
PragmaPushLocation(PragmaPushLocation) {}
};
void Act(SourceLocation PragmaLocation,
PragmaMsStackAction Action,
llvm::StringRef StackSlotLabel,
ValueType Value);

void SentinelAction(PragmaMsStackAction Action, StringRef Label) {
assert((Action == PSK_Push || Action == PSK_Pop) &&
"Can only push / pop #pragma stack sentinels!");
Act(CurrentPragmaLocation, Action, Label, CurrentValue);
}

explicit PragmaStack(const ValueType &Default)
: DefaultValue(Default), CurrentValue(Default) {}

bool hasValue() const { return CurrentValue != DefaultValue; }

SmallVector<Slot, 2> Stack;
ValueType DefaultValue; 
ValueType CurrentValue;
SourceLocation CurrentPragmaLocation;
};

PragmaStack<MSVtorDispAttr::Mode> VtorDispStack;
static const unsigned kMac68kAlignmentSentinel = ~0U;
PragmaStack<unsigned> PackStack;
struct PackIncludeState {
unsigned CurrentValue;
SourceLocation CurrentPragmaLocation;
bool HasNonDefaultValue, ShouldWarnOnInclude;
};
SmallVector<PackIncludeState, 8> PackIncludeStack;
PragmaStack<StringLiteral *> DataSegStack;
PragmaStack<StringLiteral *> BSSSegStack;
PragmaStack<StringLiteral *> ConstSegStack;
PragmaStack<StringLiteral *> CodeSegStack;

class PragmaStackSentinelRAII {
public:
PragmaStackSentinelRAII(Sema &S, StringRef SlotLabel, bool ShouldAct);
~PragmaStackSentinelRAII();

private:
Sema &S;
StringRef SlotLabel;
bool ShouldAct;
};

FileNullabilityMap NullabilityMap;

StringLiteral *CurInitSeg;
SourceLocation CurInitSegLoc;

void *VisContext; 

struct PragmaAttributeEntry {
SourceLocation Loc;
ParsedAttr *Attribute;
SmallVector<attr::SubjectMatchRule, 4> MatchRules;
bool IsUsed;
};
SmallVector<PragmaAttributeEntry, 2> PragmaAttributeStack;

const Decl *PragmaAttributeCurrentTargetDecl;

SourceLocation OptimizeOffPragmaLocation;

bool IsBuildingRecoveryCallExpr;

CleanupInfo Cleanup;

SmallVector<BlockDecl*, 8> ExprCleanupObjects;

llvm::SmallPtrSet<Expr*, 2> MaybeODRUseExprs;

std::unique_ptr<sema::FunctionScopeInfo> PreallocatedFunctionScope;

SmallVector<sema::FunctionScopeInfo *, 4> FunctionScopes;

typedef LazyVector<TypedefNameDecl *, ExternalSemaSource,
&ExternalSemaSource::ReadExtVectorDecls, 2, 2>
ExtVectorDeclsType;

ExtVectorDeclsType ExtVectorDecls;

std::unique_ptr<CXXFieldCollector> FieldCollector;

typedef llvm::SmallSetVector<NamedDecl *, 16> NamedDeclSetType;

NamedDeclSetType UnusedPrivateFields;

llvm::SmallSetVector<const TypedefNameDecl *, 4>
UnusedLocalTypedefNameCandidates;

typedef std::pair<SourceLocation, bool> DeleteExprLoc;
typedef llvm::SmallVector<DeleteExprLoc, 4> DeleteLocs;
llvm::MapVector<FieldDecl *, DeleteLocs> DeleteExprs;

typedef llvm::SmallPtrSet<const CXXRecordDecl*, 8> RecordDeclSetTy;

std::unique_ptr<RecordDeclSetTy> PureVirtualClassDiagSet;

llvm::SmallPtrSet<const Decl*, 4> ParsingInitForAutoVars;

NamedDecl *findLocallyScopedExternCDecl(DeclarationName Name);

typedef LazyVector<VarDecl *, ExternalSemaSource,
&ExternalSemaSource::ReadTentativeDefinitions, 2, 2>
TentativeDefinitionsType;

TentativeDefinitionsType TentativeDefinitions;

typedef LazyVector<const DeclaratorDecl *, ExternalSemaSource,
&ExternalSemaSource::ReadUnusedFileScopedDecls, 2, 2>
UnusedFileScopedDeclsType;

UnusedFileScopedDeclsType UnusedFileScopedDecls;

typedef LazyVector<CXXConstructorDecl *, ExternalSemaSource,
&ExternalSemaSource::ReadDelegatingConstructors, 2, 2>
DelegatingCtorDeclsType;

DelegatingCtorDeclsType DelegatingCtorDecls;

SmallVector<std::pair<const CXXMethodDecl*, const CXXMethodDecl*>, 2>
DelayedExceptionSpecChecks;

SmallVector<std::pair<CXXMethodDecl*, const FunctionProtoType*>, 2>
DelayedDefaultedMemberExceptionSpecs;

typedef llvm::MapVector<const FunctionDecl *,
std::unique_ptr<LateParsedTemplate>>
LateParsedTemplateMapT;
LateParsedTemplateMapT LateParsedTemplateMap;

typedef void LateTemplateParserCB(void *P, LateParsedTemplate &LPT);
typedef void LateTemplateParserCleanupCB(void *P);
LateTemplateParserCB *LateTemplateParser;
LateTemplateParserCleanupCB *LateTemplateParserCleanup;
void *OpaqueParser;

void SetLateTemplateParser(LateTemplateParserCB *LTP,
LateTemplateParserCleanupCB *LTPCleanup,
void *P) {
LateTemplateParser = LTP;
LateTemplateParserCleanup = LTPCleanup;
OpaqueParser = P;
}

class DelayedDiagnostics;

class DelayedDiagnosticsState {
sema::DelayedDiagnosticPool *SavedPool;
friend class Sema::DelayedDiagnostics;
};
typedef DelayedDiagnosticsState ParsingDeclState;
typedef DelayedDiagnosticsState ProcessingContextState;

class DelayedDiagnostics {
sema::DelayedDiagnosticPool *CurPool;

public:
DelayedDiagnostics() : CurPool(nullptr) {}

void add(const sema::DelayedDiagnostic &diag); 

bool shouldDelayDiagnostics() { return CurPool != nullptr; }

sema::DelayedDiagnosticPool *getCurrentPool() const {
return CurPool;
}

DelayedDiagnosticsState push(sema::DelayedDiagnosticPool &pool) {
DelayedDiagnosticsState state;
state.SavedPool = CurPool;
CurPool = &pool;
return state;
}

void popWithoutEmitting(DelayedDiagnosticsState state) {
CurPool = state.SavedPool;
}

DelayedDiagnosticsState pushUndelayed() {
DelayedDiagnosticsState state;
state.SavedPool = CurPool;
CurPool = nullptr;
return state;
}

void popUndelayed(DelayedDiagnosticsState state) {
assert(CurPool == nullptr);
CurPool = state.SavedPool;
}
} DelayedDiagnostics;

class ContextRAII {
private:
Sema &S;
DeclContext *SavedContext;
ProcessingContextState SavedContextState;
QualType SavedCXXThisTypeOverride;

public:
ContextRAII(Sema &S, DeclContext *ContextToPush, bool NewThisContext = true)
: S(S), SavedContext(S.CurContext),
SavedContextState(S.DelayedDiagnostics.pushUndelayed()),
SavedCXXThisTypeOverride(S.CXXThisTypeOverride)
{
assert(ContextToPush && "pushing null context");
S.CurContext = ContextToPush;
if (NewThisContext)
S.CXXThisTypeOverride = QualType();
}

void pop() {
if (!SavedContext) return;
S.CurContext = SavedContext;
S.DelayedDiagnostics.popUndelayed(SavedContextState);
S.CXXThisTypeOverride = SavedCXXThisTypeOverride;
SavedContext = nullptr;
}

~ContextRAII() {
pop();
}
};

class SynthesizedFunctionScope {
Sema &S;
Sema::ContextRAII SavedContext;
bool PushedCodeSynthesisContext = false;

public:
SynthesizedFunctionScope(Sema &S, DeclContext *DC)
: S(S), SavedContext(S, DC) {
S.PushFunctionScope();
S.PushExpressionEvaluationContext(
Sema::ExpressionEvaluationContext::PotentiallyEvaluated);
if (auto *FD = dyn_cast<FunctionDecl>(DC))
FD->setWillHaveBody(true);
else
assert(isa<ObjCMethodDecl>(DC));
}

void addContextNote(SourceLocation UseLoc) {
assert(!PushedCodeSynthesisContext);

Sema::CodeSynthesisContext Ctx;
Ctx.Kind = Sema::CodeSynthesisContext::DefiningSynthesizedFunction;
Ctx.PointOfInstantiation = UseLoc;
Ctx.Entity = cast<Decl>(S.CurContext);
S.pushCodeSynthesisContext(Ctx);

PushedCodeSynthesisContext = true;
}

~SynthesizedFunctionScope() {
if (PushedCodeSynthesisContext)
S.popCodeSynthesisContext();
if (auto *FD = dyn_cast<FunctionDecl>(S.CurContext))
FD->setWillHaveBody(false);
S.PopExpressionEvaluationContext();
S.PopFunctionScopeInfo();
}
};

llvm::MapVector<IdentifierInfo *, WeakInfo> WeakUndeclaredIdentifiers;

llvm::DenseMap<IdentifierInfo*,AsmLabelAttr*> ExtnameUndeclaredIdentifiers;


void LoadExternalWeakUndeclaredIdentifiers();

SmallVector<Decl*,2> WeakTopLevelDecl;

IdentifierResolver IdResolver;

Scope *TUScope;

LazyDeclPtr StdNamespace;

LazyDeclPtr StdBadAlloc;

LazyDeclPtr StdAlignValT;

NamespaceDecl *StdExperimentalNamespaceCache;

ClassTemplateDecl *StdInitializerList;

ClassTemplateDecl *StdCoroutineTraitsCache;

RecordDecl *CXXTypeInfoDecl;

RecordDecl *MSVCGuidDecl;

std::unique_ptr<NSAPI> NSAPIObj;

ObjCInterfaceDecl *NSNumberDecl;

ObjCInterfaceDecl *NSValueDecl;

QualType NSNumberPointer;

QualType NSValuePointer;

ObjCMethodDecl *NSNumberLiteralMethods[NSAPI::NumNSNumberLiteralMethods];

ObjCInterfaceDecl *NSStringDecl;

QualType NSStringPointer;

ObjCMethodDecl *StringWithUTF8StringMethod;

ObjCMethodDecl *ValueWithBytesObjCTypeMethod;

ObjCInterfaceDecl *NSArrayDecl;

ObjCMethodDecl *ArrayWithObjectsMethod;

ObjCInterfaceDecl *NSDictionaryDecl;

ObjCMethodDecl *DictionaryWithObjectsMethod;

QualType QIDNSCopying;

Selector RespondsToSelectorSel;

bool GlobalNewDeleteDeclared;

bool AllowAbstractFieldReference;

enum class ExpressionEvaluationContext {
Unevaluated,

UnevaluatedList,

DiscardedStatement,

UnevaluatedAbstract,

ConstantEvaluated,

PotentiallyEvaluated,

PotentiallyEvaluatedIfUsed
};

struct ExpressionEvaluationContextRecord {
ExpressionEvaluationContext Context;

CleanupInfo ParentCleanup;

bool IsDecltype;

unsigned NumCleanupObjects;

unsigned NumTypos;

llvm::SmallPtrSet<Expr*, 2> SavedMaybeODRUseExprs;

SmallVector<LambdaExpr *, 2> Lambdas;

Decl *ManglingContextDecl;

std::unique_ptr<MangleNumberingContext> MangleNumbering;

SmallVector<CallExpr *, 8> DelayedDecltypeCalls;

SmallVector<CXXBindTemporaryExpr *, 8> DelayedDecltypeBinds;

enum ExpressionKind {
EK_Decltype, EK_TemplateArgument, EK_Other
} ExprContext;

ExpressionEvaluationContextRecord(ExpressionEvaluationContext Context,
unsigned NumCleanupObjects,
CleanupInfo ParentCleanup,
Decl *ManglingContextDecl,
ExpressionKind ExprContext)
: Context(Context), ParentCleanup(ParentCleanup),
NumCleanupObjects(NumCleanupObjects), NumTypos(0),
ManglingContextDecl(ManglingContextDecl), MangleNumbering(),
ExprContext(ExprContext) {}

MangleNumberingContext &getMangleNumberingContext(ASTContext &Ctx);

bool isUnevaluated() const {
return Context == ExpressionEvaluationContext::Unevaluated ||
Context == ExpressionEvaluationContext::UnevaluatedAbstract ||
Context == ExpressionEvaluationContext::UnevaluatedList;
}
bool isConstantEvaluated() const {
return Context == ExpressionEvaluationContext::ConstantEvaluated;
}
};

SmallVector<ExpressionEvaluationContextRecord, 8> ExprEvalContexts;

MangleNumberingContext *getCurrentMangleNumberContext(
const DeclContext *DC,
Decl *&ManglingContextDecl);


class SpecialMemberOverloadResult {
public:
enum Kind {
NoMemberOrDeleted,
Ambiguous,
Success
};

private:
llvm::PointerIntPair<CXXMethodDecl*, 2> Pair;

public:
SpecialMemberOverloadResult() : Pair() {}
SpecialMemberOverloadResult(CXXMethodDecl *MD)
: Pair(MD, MD->isDeleted() ? NoMemberOrDeleted : Success) {}

CXXMethodDecl *getMethod() const { return Pair.getPointer(); }
void setMethod(CXXMethodDecl *MD) { Pair.setPointer(MD); }

Kind getKind() const { return static_cast<Kind>(Pair.getInt()); }
void setKind(Kind K) { Pair.setInt(K); }
};

class SpecialMemberOverloadResultEntry
: public llvm::FastFoldingSetNode,
public SpecialMemberOverloadResult {
public:
SpecialMemberOverloadResultEntry(const llvm::FoldingSetNodeID &ID)
: FastFoldingSetNode(ID)
{}
};

llvm::FoldingSet<SpecialMemberOverloadResultEntry> SpecialMemberCache;

mutable llvm::DenseMap<const EnumDecl*, llvm::APInt> FlagBitsCache;

TranslationUnitKind TUKind;

llvm::BumpPtrAllocator BumpAlloc;

unsigned NumSFINAEErrors;

typedef llvm::DenseMap<ParmVarDecl *, llvm::TinyPtrVector<ParmVarDecl *>>
UnparsedDefaultArgInstantiationsMap;

UnparsedDefaultArgInstantiationsMap UnparsedDefaultArgInstantiations;

llvm::DenseMap<ParmVarDecl *, SourceLocation> UnparsedDefaultArgLocs;

llvm::MapVector<NamedDecl *, SourceLocation> UndefinedButUsed;

bool isExternalWithNoLinkageType(ValueDecl *VD);

void getUndefinedButUsed(
SmallVectorImpl<std::pair<NamedDecl *, SourceLocation> > &Undefined);

const llvm::MapVector<FieldDecl *, DeleteLocs> &
getMismatchingDeleteExpressions() const;

typedef std::pair<ObjCMethodList, ObjCMethodList> GlobalMethods;
typedef llvm::DenseMap<Selector, GlobalMethods> GlobalMethodPool;

GlobalMethodPool MethodPool;

llvm::MapVector<Selector, SourceLocation> ReferencedSelectors;

enum CXXSpecialMember {
CXXDefaultConstructor,
CXXCopyConstructor,
CXXMoveConstructor,
CXXCopyAssignment,
CXXMoveAssignment,
CXXDestructor,
CXXInvalid
};

typedef llvm::PointerIntPair<CXXRecordDecl *, 3, CXXSpecialMember>
SpecialMemberDecl;

llvm::SmallPtrSet<SpecialMemberDecl, 4> SpecialMembersBeingDeclared;

llvm::SmallPtrSet<const NamedDecl *, 4> TypoCorrectedFunctionDefinitions;

llvm::SmallVector<QualType, 4> CurrentParameterCopyTypes;

void ReadMethodPool(Selector Sel);
void updateOutOfDateSelector(Selector Sel);

bool isSelfExpr(Expr *RExpr);
bool isSelfExpr(Expr *RExpr, const ObjCMethodDecl *Method);

void EmitCurrentDiagnostic(unsigned DiagID);

class FPContractStateRAII {
public:
FPContractStateRAII(Sema &S) : S(S), OldFPFeaturesState(S.FPFeatures) {}
~FPContractStateRAII() { S.FPFeatures = OldFPFeaturesState; }

private:
Sema& S;
FPOptions OldFPFeaturesState;
};

void addImplicitTypedef(StringRef Name, QualType T);

public:
Sema(Preprocessor &pp, ASTContext &ctxt, ASTConsumer &consumer,
TranslationUnitKind TUKind = TU_Complete,
CodeCompleteConsumer *CompletionConsumer = nullptr);
~Sema();

void Initialize();

const LangOptions &getLangOpts() const { return LangOpts; }
OpenCLOptions &getOpenCLOptions() { return OpenCLFeatures; }
FPOptions     &getFPOptions() { return FPFeatures; }

DiagnosticsEngine &getDiagnostics() const { return Diags; }
SourceManager &getSourceManager() const { return SourceMgr; }
Preprocessor &getPreprocessor() const { return PP; }
ASTContext &getASTContext() const { return Context; }
ASTConsumer &getASTConsumer() const { return Consumer; }
ASTMutationListener *getASTMutationListener() const;
ExternalSemaSource* getExternalSource() const { return ExternalSource; }

void addExternalSource(ExternalSemaSource *E);

void PrintStats() const;

class SemaDiagnosticBuilder : public DiagnosticBuilder {
Sema &SemaRef;
unsigned DiagID;

public:
SemaDiagnosticBuilder(DiagnosticBuilder &DB, Sema &SemaRef, unsigned DiagID)
: DiagnosticBuilder(DB), SemaRef(SemaRef), DiagID(DiagID) { }

SemaDiagnosticBuilder(const SemaDiagnosticBuilder&) = default;

~SemaDiagnosticBuilder() {
if (!isActive()) return;

FlushCounts();
Clear();

SemaRef.EmitCurrentDiagnostic(DiagID);
}

template<typename T>
friend const SemaDiagnosticBuilder &operator<<(
const SemaDiagnosticBuilder &Diag, const T &Value) {
const DiagnosticBuilder &BaseDiag = Diag;
BaseDiag << Value;
return Diag;
}
};

SemaDiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID) {
DiagnosticBuilder DB = Diags.Report(Loc, DiagID);
return SemaDiagnosticBuilder(DB, *this, DiagID);
}

SemaDiagnosticBuilder Diag(SourceLocation Loc, const PartialDiagnostic& PD);

PartialDiagnostic PDiag(unsigned DiagID = 0); 

bool findMacroSpelling(SourceLocation &loc, StringRef name);

std::string
getFixItZeroInitializerForType(QualType T, SourceLocation Loc) const;
std::string getFixItZeroLiteralForType(QualType T, SourceLocation Loc) const;

SourceLocation getLocForEndOfToken(SourceLocation Loc, unsigned Offset = 0);

ModuleLoader &getModuleLoader() const;

void emitAndClearUnusedLocalTypedefWarnings();

void ActOnStartOfTranslationUnit();
void ActOnEndOfTranslationUnit();

void CheckDelegatingCtorCycles();

Scope *getScopeForContext(DeclContext *Ctx);

void PushFunctionScope();
void PushBlockScope(Scope *BlockScope, BlockDecl *Block);
sema::LambdaScopeInfo *PushLambdaScope();

void RecordParsingTemplateParameterDepth(unsigned Depth);

void PushCapturedRegionScope(Scope *RegionScope, CapturedDecl *CD,
RecordDecl *RD,
CapturedRegionKind K);
void
PopFunctionScopeInfo(const sema::AnalysisBasedWarnings::Policy *WP = nullptr,
const Decl *D = nullptr,
const BlockExpr *blkExpr = nullptr);

sema::FunctionScopeInfo *getCurFunction() const {
return FunctionScopes.empty() ? nullptr : FunctionScopes.back();
}

sema::FunctionScopeInfo *getEnclosingFunction() const;

void setFunctionHasBranchIntoScope();
void setFunctionHasBranchProtectedScope();
void setFunctionHasIndirectGoto();

void PushCompoundScope(bool IsStmtExpr);
void PopCompoundScope();

sema::CompoundScopeInfo &getCurCompoundScope() const;

bool hasAnyUnrecoverableErrorsInThisFunction() const;

sema::BlockScopeInfo *getCurBlock();

sema::LambdaScopeInfo *
getCurLambda(bool IgnoreNonLambdaCapturingScope = false);

sema::LambdaScopeInfo *getCurGenericLambda();

sema::CapturedRegionScopeInfo *getCurCapturedRegion();

SmallVectorImpl<Decl *> &WeakTopLevelDecls() { return WeakTopLevelDecl; }

void ActOnComment(SourceRange Comment);


QualType BuildQualifiedType(QualType T, SourceLocation Loc, Qualifiers Qs,
const DeclSpec *DS = nullptr);
QualType BuildQualifiedType(QualType T, SourceLocation Loc, unsigned CVRA,
const DeclSpec *DS = nullptr);
QualType BuildPointerType(QualType T,
SourceLocation Loc, DeclarationName Entity);
QualType BuildReferenceType(QualType T, bool LValueRef,
SourceLocation Loc, DeclarationName Entity);
QualType BuildArrayType(QualType T, ArrayType::ArraySizeModifier ASM,
Expr *ArraySize, unsigned Quals,
SourceRange Brackets, DeclarationName Entity);
QualType BuildVectorType(QualType T, Expr *VecSize, SourceLocation AttrLoc);
QualType BuildExtVectorType(QualType T, Expr *ArraySize,
SourceLocation AttrLoc);
QualType BuildAddressSpaceAttr(QualType &T, Expr *AddrSpace,
SourceLocation AttrLoc);

bool CheckFunctionReturnType(QualType T, SourceLocation Loc);

QualType BuildFunctionType(QualType T,
MutableArrayRef<QualType> ParamTypes,
SourceLocation Loc, DeclarationName Entity,
const FunctionProtoType::ExtProtoInfo &EPI);

QualType BuildMemberPointerType(QualType T, QualType Class,
SourceLocation Loc,
DeclarationName Entity);
QualType BuildBlockPointerType(QualType T,
SourceLocation Loc, DeclarationName Entity);
QualType BuildParenType(QualType T);
QualType BuildAtomicType(QualType T, SourceLocation Loc);
QualType BuildReadPipeType(QualType T,
SourceLocation Loc);
QualType BuildWritePipeType(QualType T,
SourceLocation Loc);

TypeSourceInfo *GetTypeForDeclarator(Declarator &D, Scope *S);
TypeSourceInfo *GetTypeForDeclaratorCast(Declarator &D, QualType FromTy);
TypeSourceInfo *GetTypeSourceInfoForDeclarator(Declarator &D, QualType T,
TypeSourceInfo *ReturnTypeInfo);

ParsedType CreateParsedType(QualType T, TypeSourceInfo *TInfo);
DeclarationNameInfo GetNameForDeclarator(Declarator &D);
DeclarationNameInfo GetNameFromUnqualifiedId(const UnqualifiedId &Name);
static QualType GetTypeFromParser(ParsedType Ty,
TypeSourceInfo **TInfo = nullptr);
CanThrowResult canThrow(const Expr *E);
const FunctionProtoType *ResolveExceptionSpec(SourceLocation Loc,
const FunctionProtoType *FPT);
void UpdateExceptionSpec(FunctionDecl *FD,
const FunctionProtoType::ExceptionSpecInfo &ESI);
bool CheckSpecifiedExceptionType(QualType &T, SourceRange Range);
bool CheckDistantExceptionSpec(QualType T);
bool CheckEquivalentExceptionSpec(FunctionDecl *Old, FunctionDecl *New);
bool CheckEquivalentExceptionSpec(
const FunctionProtoType *Old, SourceLocation OldLoc,
const FunctionProtoType *New, SourceLocation NewLoc);
bool CheckEquivalentExceptionSpec(
const PartialDiagnostic &DiagID, const PartialDiagnostic & NoteID,
const FunctionProtoType *Old, SourceLocation OldLoc,
const FunctionProtoType *New, SourceLocation NewLoc);
bool handlerCanCatch(QualType HandlerType, QualType ExceptionType);
bool CheckExceptionSpecSubset(const PartialDiagnostic &DiagID,
const PartialDiagnostic &NestedDiagID,
const PartialDiagnostic &NoteID,
const FunctionProtoType *Superset,
SourceLocation SuperLoc,
const FunctionProtoType *Subset,
SourceLocation SubLoc);
bool CheckParamExceptionSpec(const PartialDiagnostic &NestedDiagID,
const PartialDiagnostic &NoteID,
const FunctionProtoType *Target,
SourceLocation TargetLoc,
const FunctionProtoType *Source,
SourceLocation SourceLoc);

TypeResult ActOnTypeName(Scope *S, Declarator &D);

ParsedType ActOnObjCInstanceType(SourceLocation Loc);

struct TypeDiagnoser {
TypeDiagnoser() {}

virtual void diagnose(Sema &S, SourceLocation Loc, QualType T) = 0;
virtual ~TypeDiagnoser() {}
};

static int getPrintable(int I) { return I; }
static unsigned getPrintable(unsigned I) { return I; }
static bool getPrintable(bool B) { return B; }
static const char * getPrintable(const char *S) { return S; }
static StringRef getPrintable(StringRef S) { return S; }
static const std::string &getPrintable(const std::string &S) { return S; }
static const IdentifierInfo *getPrintable(const IdentifierInfo *II) {
return II;
}
static DeclarationName getPrintable(DeclarationName N) { return N; }
static QualType getPrintable(QualType T) { return T; }
static SourceRange getPrintable(SourceRange R) { return R; }
static SourceRange getPrintable(SourceLocation L) { return L; }
static SourceRange getPrintable(const Expr *E) { return E->getSourceRange(); }
static SourceRange getPrintable(TypeLoc TL) { return TL.getSourceRange();}

template <typename... Ts> class BoundTypeDiagnoser : public TypeDiagnoser {
unsigned DiagID;
std::tuple<const Ts &...> Args;

template <std::size_t... Is>
void emit(const SemaDiagnosticBuilder &DB,
llvm::index_sequence<Is...>) const {
bool Dummy[] = {false, (DB << getPrintable(std::get<Is>(Args)))...};
(void)Dummy;
}

public:
BoundTypeDiagnoser(unsigned DiagID, const Ts &...Args)
: TypeDiagnoser(), DiagID(DiagID), Args(Args...) {
assert(DiagID != 0 && "no diagnostic for type diagnoser");
}

void diagnose(Sema &S, SourceLocation Loc, QualType T) override {
const SemaDiagnosticBuilder &DB = S.Diag(Loc, DiagID);
emit(DB, llvm::index_sequence_for<Ts...>());
DB << T;
}
};

private:
bool RequireCompleteTypeImpl(SourceLocation Loc, QualType T,
TypeDiagnoser *Diagnoser);

struct ModuleScope {
clang::Module *Module = nullptr;
bool ModuleInterface = false;
VisibleModuleSet OuterVisibleModules;
};
llvm::SmallVector<ModuleScope, 16> ModuleScopes;

Module *getCurrentModule() const {
return ModuleScopes.empty() ? nullptr : ModuleScopes.back().Module;
}

VisibleModuleSet VisibleModules;

public:
Module *getOwningModule(Decl *Entity) { return Entity->getOwningModule(); }

void makeMergedDefinitionVisible(NamedDecl *ND);

bool isModuleVisible(const Module *M) { return VisibleModules.isVisible(M); }

bool isVisible(const NamedDecl *D) {
return !D->isHidden() || isVisibleSlow(D);
}

bool
hasVisibleDeclaration(const NamedDecl *D,
llvm::SmallVectorImpl<Module *> *Modules = nullptr) {
return isVisible(D) || hasVisibleDeclarationSlow(D, Modules);
}
bool hasVisibleDeclarationSlow(const NamedDecl *D,
llvm::SmallVectorImpl<Module *> *Modules);

bool hasVisibleMergedDefinition(NamedDecl *Def);
bool hasMergedDefinitionInCurrentModule(NamedDecl *Def);

bool hasStructuralCompatLayout(Decl *D, Decl *Suggested);

bool hasVisibleDefinition(NamedDecl *D, NamedDecl **Suggested,
bool OnlyNeedComplete = false);
bool hasVisibleDefinition(const NamedDecl *D) {
NamedDecl *Hidden;
return hasVisibleDefinition(const_cast<NamedDecl*>(D), &Hidden);
}

bool
hasVisibleDefaultArgument(const NamedDecl *D,
llvm::SmallVectorImpl<Module *> *Modules = nullptr);

bool hasVisibleExplicitSpecialization(
const NamedDecl *D, llvm::SmallVectorImpl<Module *> *Modules = nullptr);

bool hasVisibleMemberSpecialization(
const NamedDecl *D, llvm::SmallVectorImpl<Module *> *Modules = nullptr);

bool isEquivalentInternalLinkageDeclaration(const NamedDecl *A,
const NamedDecl *B);
void diagnoseEquivalentInternalLinkageDeclarations(
SourceLocation Loc, const NamedDecl *D,
ArrayRef<const NamedDecl *> Equiv);

bool isCompleteType(SourceLocation Loc, QualType T) {
return !RequireCompleteTypeImpl(Loc, T, nullptr);
}
bool RequireCompleteType(SourceLocation Loc, QualType T,
TypeDiagnoser &Diagnoser);
bool RequireCompleteType(SourceLocation Loc, QualType T,
unsigned DiagID);

template <typename... Ts>
bool RequireCompleteType(SourceLocation Loc, QualType T, unsigned DiagID,
const Ts &...Args) {
BoundTypeDiagnoser<Ts...> Diagnoser(DiagID, Args...);
return RequireCompleteType(Loc, T, Diagnoser);
}

void completeExprArrayBound(Expr *E);
bool RequireCompleteExprType(Expr *E, TypeDiagnoser &Diagnoser);
bool RequireCompleteExprType(Expr *E, unsigned DiagID);

template <typename... Ts>
bool RequireCompleteExprType(Expr *E, unsigned DiagID, const Ts &...Args) {
BoundTypeDiagnoser<Ts...> Diagnoser(DiagID, Args...);
return RequireCompleteExprType(E, Diagnoser);
}

bool RequireLiteralType(SourceLocation Loc, QualType T,
TypeDiagnoser &Diagnoser);
bool RequireLiteralType(SourceLocation Loc, QualType T, unsigned DiagID);

template <typename... Ts>
bool RequireLiteralType(SourceLocation Loc, QualType T, unsigned DiagID,
const Ts &...Args) {
BoundTypeDiagnoser<Ts...> Diagnoser(DiagID, Args...);
return RequireLiteralType(Loc, T, Diagnoser);
}

QualType getElaboratedType(ElaboratedTypeKeyword Keyword,
const CXXScopeSpec &SS, QualType T,
TagDecl *OwnedTagDecl = nullptr);

QualType BuildTypeofExprType(Expr *E, SourceLocation Loc);
QualType BuildDecltypeType(Expr *E, SourceLocation Loc,
bool AsUnevaluated = true);
QualType BuildUnaryTransformType(QualType BaseType,
UnaryTransformType::UTTKind UKind,
SourceLocation Loc);


struct SkipBodyInfo {
SkipBodyInfo()
: ShouldSkip(false), CheckSameAsPrevious(false), Previous(nullptr),
New(nullptr) {}
bool ShouldSkip;
bool CheckSameAsPrevious;
NamedDecl *Previous;
NamedDecl *New;
};

DeclGroupPtrTy ConvertDeclToDeclGroup(Decl *Ptr, Decl *OwnedType = nullptr);

void DiagnoseUseOfUnimplementedSelectors();

bool isSimpleTypeSpecifier(tok::TokenKind Kind) const;

ParsedType getTypeName(const IdentifierInfo &II, SourceLocation NameLoc,
Scope *S, CXXScopeSpec *SS = nullptr,
bool isClassName = false, bool HasTrailingDot = false,
ParsedType ObjectType = nullptr,
bool IsCtorOrDtorName = false,
bool WantNontrivialTypeSourceInfo = false,
bool IsClassTemplateDeductionContext = true,
IdentifierInfo **CorrectedII = nullptr);
TypeSpecifierType isTagName(IdentifierInfo &II, Scope *S);
bool isMicrosoftMissingTypename(const CXXScopeSpec *SS, Scope *S);
void DiagnoseUnknownTypeName(IdentifierInfo *&II,
SourceLocation IILoc,
Scope *S,
CXXScopeSpec *SS,
ParsedType &SuggestedType,
bool IsTemplateName = false);

ParsedType ActOnMSVCUnknownTypeName(const IdentifierInfo &II,
SourceLocation NameLoc,
bool IsTemplateTypeArg);

enum NameClassificationKind {
NC_Unknown,
NC_Error,
NC_Keyword,
NC_Type,
NC_Expression,
NC_NestedNameSpecifier,
NC_TypeTemplate,
NC_VarTemplate,
NC_FunctionTemplate
};

class NameClassification {
NameClassificationKind Kind;
ExprResult Expr;
TemplateName Template;
ParsedType Type;

explicit NameClassification(NameClassificationKind Kind) : Kind(Kind) {}

public:
NameClassification(ExprResult Expr) : Kind(NC_Expression), Expr(Expr) {}

NameClassification(ParsedType Type) : Kind(NC_Type), Type(Type) {}

NameClassification(const IdentifierInfo *Keyword) : Kind(NC_Keyword) {}

static NameClassification Error() {
return NameClassification(NC_Error);
}

static NameClassification Unknown() {
return NameClassification(NC_Unknown);
}

static NameClassification NestedNameSpecifier() {
return NameClassification(NC_NestedNameSpecifier);
}

static NameClassification TypeTemplate(TemplateName Name) {
NameClassification Result(NC_TypeTemplate);
Result.Template = Name;
return Result;
}

static NameClassification VarTemplate(TemplateName Name) {
NameClassification Result(NC_VarTemplate);
Result.Template = Name;
return Result;
}

static NameClassification FunctionTemplate(TemplateName Name) {
NameClassification Result(NC_FunctionTemplate);
Result.Template = Name;
return Result;
}

NameClassificationKind getKind() const { return Kind; }

ParsedType getType() const {
assert(Kind == NC_Type);
return Type;
}

ExprResult getExpression() const {
assert(Kind == NC_Expression);
return Expr;
}

TemplateName getTemplateName() const {
assert(Kind == NC_TypeTemplate || Kind == NC_FunctionTemplate ||
Kind == NC_VarTemplate);
return Template;
}

TemplateNameKind getTemplateNameKind() const {
switch (Kind) {
case NC_TypeTemplate:
return TNK_Type_template;
case NC_FunctionTemplate:
return TNK_Function_template;
case NC_VarTemplate:
return TNK_Var_template;
default:
llvm_unreachable("unsupported name classification.");
}
}
};

NameClassification
ClassifyName(Scope *S, CXXScopeSpec &SS, IdentifierInfo *&Name,
SourceLocation NameLoc, const Token &NextToken,
bool IsAddressOfOperand,
std::unique_ptr<CorrectionCandidateCallback> CCC = nullptr);

enum class TemplateNameKindForDiagnostics {
ClassTemplate,
FunctionTemplate,
VarTemplate,
AliasTemplate,
TemplateTemplateParam,
DependentTemplate
};
TemplateNameKindForDiagnostics
getTemplateNameKindForDiagnostics(TemplateName Name);

bool mightBeIntendedToBeTemplateName(ExprResult E, bool &Dependent) {
if (!getLangOpts().CPlusPlus || E.isInvalid())
return false;
Dependent = false;
if (auto *DRE = dyn_cast<DeclRefExpr>(E.get()))
return !DRE->hasExplicitTemplateArgs();
if (auto *ME = dyn_cast<MemberExpr>(E.get()))
return !ME->hasExplicitTemplateArgs();
Dependent = true;
if (auto *DSDRE = dyn_cast<DependentScopeDeclRefExpr>(E.get()))
return !DSDRE->hasExplicitTemplateArgs();
if (auto *DSME = dyn_cast<CXXDependentScopeMemberExpr>(E.get()))
return !DSME->hasExplicitTemplateArgs();
return false;
}
void diagnoseExprIntendedAsTemplateName(Scope *S, ExprResult TemplateName,
SourceLocation Less,
SourceLocation Greater);

Decl *ActOnDeclarator(Scope *S, Declarator &D);

NamedDecl *HandleDeclarator(Scope *S, Declarator &D,
MultiTemplateParamsArg TemplateParameterLists);
void RegisterLocallyScopedExternCDecl(NamedDecl *ND, Scope *S);
bool DiagnoseClassNameShadow(DeclContext *DC, DeclarationNameInfo Info);
bool diagnoseQualifiedDeclaration(CXXScopeSpec &SS, DeclContext *DC,
DeclarationName Name, SourceLocation Loc,
bool IsTemplateId);
void
diagnoseIgnoredQualifiers(unsigned DiagID, unsigned Quals,
SourceLocation FallbackLoc,
SourceLocation ConstQualLoc = SourceLocation(),
SourceLocation VolatileQualLoc = SourceLocation(),
SourceLocation RestrictQualLoc = SourceLocation(),
SourceLocation AtomicQualLoc = SourceLocation(),
SourceLocation UnalignedQualLoc = SourceLocation());

static bool adjustContextForLocalExternDecl(DeclContext *&DC);
void DiagnoseFunctionSpecifiers(const DeclSpec &DS);
NamedDecl *getShadowedDeclaration(const TypedefNameDecl *D,
const LookupResult &R);
NamedDecl *getShadowedDeclaration(const VarDecl *D, const LookupResult &R);
void CheckShadow(NamedDecl *D, NamedDecl *ShadowedDecl,
const LookupResult &R);
void CheckShadow(Scope *S, VarDecl *D);

void CheckShadowingDeclModification(Expr *E, SourceLocation Loc);

void DiagnoseShadowingLambdaDecls(const sema::LambdaScopeInfo *LSI);

private:
llvm::DenseMap<const NamedDecl *, const NamedDecl *> ShadowingDecls;

public:
void CheckCastAlign(Expr *Op, QualType T, SourceRange TRange);
void handleTagNumbering(const TagDecl *Tag, Scope *TagScope);
void setTagNameForLinkagePurposes(TagDecl *TagFromDeclSpec,
TypedefNameDecl *NewTD);
void CheckTypedefForVariablyModifiedType(Scope *S, TypedefNameDecl *D);
NamedDecl* ActOnTypedefDeclarator(Scope* S, Declarator& D, DeclContext* DC,
TypeSourceInfo *TInfo,
LookupResult &Previous);
NamedDecl* ActOnTypedefNameDecl(Scope* S, DeclContext* DC, TypedefNameDecl *D,
LookupResult &Previous, bool &Redeclaration);
NamedDecl *ActOnVariableDeclarator(Scope *S, Declarator &D, DeclContext *DC,
TypeSourceInfo *TInfo,
LookupResult &Previous,
MultiTemplateParamsArg TemplateParamLists,
bool &AddToScope,
ArrayRef<BindingDecl *> Bindings = None);
NamedDecl *
ActOnDecompositionDeclarator(Scope *S, Declarator &D,
MultiTemplateParamsArg TemplateParamLists);
bool CheckVariableDeclaration(VarDecl *NewVD, LookupResult &Previous);
void CheckVariableDeclarationType(VarDecl *NewVD);
bool DeduceVariableDeclarationType(VarDecl *VDecl, bool DirectInit,
Expr *Init);
void CheckCompleteVariableDeclaration(VarDecl *VD);
void CheckCompleteDecompositionDeclaration(DecompositionDecl *DD);
void MaybeSuggestAddingStaticToDecl(const FunctionDecl *D);

NamedDecl* ActOnFunctionDeclarator(Scope* S, Declarator& D, DeclContext* DC,
TypeSourceInfo *TInfo,
LookupResult &Previous,
MultiTemplateParamsArg TemplateParamLists,
bool &AddToScope);
bool AddOverriddenMethods(CXXRecordDecl *DC, CXXMethodDecl *MD);

bool CheckConstexprFunctionDecl(const FunctionDecl *FD);
bool CheckConstexprFunctionBody(const FunctionDecl *FD, Stmt *Body);

void DiagnoseHiddenVirtualMethods(CXXMethodDecl *MD);
void FindHiddenVirtualMethods(CXXMethodDecl *MD,
SmallVectorImpl<CXXMethodDecl*> &OverloadedMethods);
void NoteHiddenVirtualMethods(CXXMethodDecl *MD,
SmallVectorImpl<CXXMethodDecl*> &OverloadedMethods);
bool CheckFunctionDeclaration(Scope *S,
FunctionDecl *NewFD, LookupResult &Previous,
bool IsMemberSpecialization);
bool shouldLinkDependentDeclWithPrevious(Decl *D, Decl *OldDecl);
bool canFullyTypeCheckRedeclaration(ValueDecl *NewD, ValueDecl *OldD,
QualType NewT, QualType OldT);
void CheckMain(FunctionDecl *FD, const DeclSpec &D);
void CheckMSVCRTEntryPoint(FunctionDecl *FD);
Attr *getImplicitCodeSegOrSectionAttrForFunction(const FunctionDecl *FD, bool IsDefinition);
Decl *ActOnParamDeclarator(Scope *S, Declarator &D);
ParmVarDecl *BuildParmVarDeclForTypedef(DeclContext *DC,
SourceLocation Loc,
QualType T);
ParmVarDecl *CheckParameter(DeclContext *DC, SourceLocation StartLoc,
SourceLocation NameLoc, IdentifierInfo *Name,
QualType T, TypeSourceInfo *TSInfo,
StorageClass SC);
void ActOnParamDefaultArgument(Decl *param,
SourceLocation EqualLoc,
Expr *defarg);
void ActOnParamUnparsedDefaultArgument(Decl *param,
SourceLocation EqualLoc,
SourceLocation ArgLoc);
void ActOnParamDefaultArgumentError(Decl *param, SourceLocation EqualLoc);
bool SetParamDefaultArgument(ParmVarDecl *Param, Expr *DefaultArg,
SourceLocation EqualLoc);

void AddInitializerToDecl(Decl *dcl, Expr *init, bool DirectInit);
void ActOnUninitializedDecl(Decl *dcl);
void ActOnInitializerError(Decl *Dcl);

void ActOnPureSpecifier(Decl *D, SourceLocation PureSpecLoc);
void ActOnCXXForRangeDecl(Decl *D);
StmtResult ActOnCXXForRangeIdentifier(Scope *S, SourceLocation IdentLoc,
IdentifierInfo *Ident,
ParsedAttributes &Attrs,
SourceLocation AttrEnd);
void SetDeclDeleted(Decl *dcl, SourceLocation DelLoc);
void SetDeclDefaulted(Decl *dcl, SourceLocation DefaultLoc);
void FinalizeDeclaration(Decl *D);
DeclGroupPtrTy FinalizeDeclaratorGroup(Scope *S, const DeclSpec &DS,
ArrayRef<Decl *> Group);
DeclGroupPtrTy BuildDeclaratorGroup(MutableArrayRef<Decl *> Group);

void ActOnDocumentableDecl(Decl *D);
void ActOnDocumentableDecls(ArrayRef<Decl *> Group);

void ActOnFinishKNRParamDeclarations(Scope *S, Declarator &D,
SourceLocation LocAfterDecls);
void CheckForFunctionRedefinition(
FunctionDecl *FD, const FunctionDecl *EffectiveDefinition = nullptr,
SkipBodyInfo *SkipBody = nullptr);
Decl *ActOnStartOfFunctionDef(Scope *S, Declarator &D,
MultiTemplateParamsArg TemplateParamLists,
SkipBodyInfo *SkipBody = nullptr);
Decl *ActOnStartOfFunctionDef(Scope *S, Decl *D,
SkipBodyInfo *SkipBody = nullptr);
void ActOnStartOfObjCMethodDef(Scope *S, Decl *D);
bool isObjCMethodDecl(Decl *D) {
return D && isa<ObjCMethodDecl>(D);
}

bool canDelayFunctionBody(const Declarator &D);

bool canSkipFunctionBody(Decl *D);

void computeNRVO(Stmt *Body, sema::FunctionScopeInfo *Scope);
Decl *ActOnFinishFunctionBody(Decl *Decl, Stmt *Body);
Decl *ActOnFinishFunctionBody(Decl *Decl, Stmt *Body, bool IsInstantiation);
Decl *ActOnSkippedFunctionBody(Decl *Decl);
void ActOnFinishInlineFunctionDef(FunctionDecl *D);

void ActOnFinishDelayedAttribute(Scope *S, Decl *D, ParsedAttributes &Attrs);

void DiagnoseUnusedParameters(ArrayRef<ParmVarDecl *> Parameters);

void
DiagnoseSizeOfParametersAndReturnValue(ArrayRef<ParmVarDecl *> Parameters,
QualType ReturnTy, NamedDecl *D);

void DiagnoseInvalidJumps(Stmt *Body);
Decl *ActOnFileScopeAsmDecl(Expr *expr,
SourceLocation AsmLoc,
SourceLocation RParenLoc);

Decl *ActOnEmptyDeclaration(Scope *S, const ParsedAttributesView &AttrList,
SourceLocation SemiLoc);

enum class ModuleDeclKind {
Interface,      
Implementation, 
Partition,      
};

DeclGroupPtrTy ActOnModuleDecl(SourceLocation StartLoc,
SourceLocation ModuleLoc, ModuleDeclKind MDK,
ModuleIdPath Path);

DeclResult ActOnModuleImport(SourceLocation AtLoc, SourceLocation ImportLoc,
ModuleIdPath Path);

void ActOnModuleInclude(SourceLocation DirectiveLoc, Module *Mod);
void BuildModuleInclude(SourceLocation DirectiveLoc, Module *Mod);

void ActOnModuleBegin(SourceLocation DirectiveLoc, Module *Mod);
void ActOnModuleEnd(SourceLocation DirectiveLoc, Module *Mod);

void createImplicitModuleImportForErrorRecovery(SourceLocation Loc,
Module *Mod);

enum class MissingImportKind {
Declaration,
Definition,
DefaultArgument,
ExplicitSpecialization,
PartialSpecialization
};

void diagnoseMissingImport(SourceLocation Loc, NamedDecl *Decl,
MissingImportKind MIK, bool Recover = true);
void diagnoseMissingImport(SourceLocation Loc, NamedDecl *Decl,
SourceLocation DeclLoc, ArrayRef<Module *> Modules,
MissingImportKind MIK, bool Recover);

Decl *ActOnStartExportDecl(Scope *S, SourceLocation ExportLoc,
SourceLocation LBraceLoc);
Decl *ActOnFinishExportDecl(Scope *S, Decl *ExportDecl,
SourceLocation RBraceLoc);

void checkSpecializationVisibility(SourceLocation Loc, NamedDecl *Spec);

void checkPartialSpecializationVisibility(SourceLocation Loc,
NamedDecl *Spec);

PrintingPolicy getPrintingPolicy() const {
return getPrintingPolicy(Context, PP);
}

static PrintingPolicy getPrintingPolicy(const ASTContext &Ctx,
const Preprocessor &PP);

void ActOnPopScope(SourceLocation Loc, Scope *S);
void ActOnTranslationUnitScope(Scope *S);

Decl *ParsedFreeStandingDeclSpec(Scope *S, AccessSpecifier AS, DeclSpec &DS,
RecordDecl *&AnonRecord);
Decl *ParsedFreeStandingDeclSpec(Scope *S, AccessSpecifier AS, DeclSpec &DS,
MultiTemplateParamsArg TemplateParams,
bool IsExplicitInstantiation,
RecordDecl *&AnonRecord);

Decl *BuildAnonymousStructOrUnion(Scope *S, DeclSpec &DS,
AccessSpecifier AS,
RecordDecl *Record,
const PrintingPolicy &Policy);

Decl *BuildMicrosoftCAnonymousStruct(Scope *S, DeclSpec &DS,
RecordDecl *Record);

enum NonTagKind {
NTK_NonStruct,
NTK_NonClass,
NTK_NonUnion,
NTK_NonEnum,
NTK_Typedef,
NTK_TypeAlias,
NTK_Template,
NTK_TypeAliasTemplate,
NTK_TemplateTemplateArgument,
};

NonTagKind getNonTagTypeDeclKind(const Decl *D, TagTypeKind TTK);

bool isAcceptableTagRedeclaration(const TagDecl *Previous,
TagTypeKind NewTag, bool isDefinition,
SourceLocation NewTagLoc,
const IdentifierInfo *Name);

enum TagUseKind {
TUK_Reference,   
TUK_Declaration, 
TUK_Definition,  
TUK_Friend       
};

Decl *ActOnTag(Scope *S, unsigned TagSpec, TagUseKind TUK,
SourceLocation KWLoc, CXXScopeSpec &SS, IdentifierInfo *Name,
SourceLocation NameLoc, const ParsedAttributesView &Attr,
AccessSpecifier AS, SourceLocation ModulePrivateLoc,
MultiTemplateParamsArg TemplateParameterLists, bool &OwnedDecl,
bool &IsDependent, SourceLocation ScopedEnumKWLoc,
bool ScopedEnumUsesClassTag, TypeResult UnderlyingType,
bool IsTypeSpecifier, bool IsTemplateParamOrArg,
SkipBodyInfo *SkipBody = nullptr);

Decl *ActOnTemplatedFriendTag(Scope *S, SourceLocation FriendLoc,
unsigned TagSpec, SourceLocation TagLoc,
CXXScopeSpec &SS, IdentifierInfo *Name,
SourceLocation NameLoc,
const ParsedAttributesView &Attr,
MultiTemplateParamsArg TempParamLists);

TypeResult ActOnDependentTag(Scope *S,
unsigned TagSpec,
TagUseKind TUK,
const CXXScopeSpec &SS,
IdentifierInfo *Name,
SourceLocation TagLoc,
SourceLocation NameLoc);

void ActOnDefs(Scope *S, Decl *TagD, SourceLocation DeclStart,
IdentifierInfo *ClassName,
SmallVectorImpl<Decl *> &Decls);
Decl *ActOnField(Scope *S, Decl *TagD, SourceLocation DeclStart,
Declarator &D, Expr *BitfieldWidth);

FieldDecl *HandleField(Scope *S, RecordDecl *TagD, SourceLocation DeclStart,
Declarator &D, Expr *BitfieldWidth,
InClassInitStyle InitStyle,
AccessSpecifier AS);
MSPropertyDecl *HandleMSProperty(Scope *S, RecordDecl *TagD,
SourceLocation DeclStart, Declarator &D,
Expr *BitfieldWidth,
InClassInitStyle InitStyle,
AccessSpecifier AS,
const ParsedAttr &MSPropertyAttr);

FieldDecl *CheckFieldDecl(DeclarationName Name, QualType T,
TypeSourceInfo *TInfo,
RecordDecl *Record, SourceLocation Loc,
bool Mutable, Expr *BitfieldWidth,
InClassInitStyle InitStyle,
SourceLocation TSSL,
AccessSpecifier AS, NamedDecl *PrevDecl,
Declarator *D = nullptr);

bool CheckNontrivialField(FieldDecl *FD);
void DiagnoseNontrivial(const CXXRecordDecl *Record, CXXSpecialMember CSM);

enum TrivialABIHandling {
TAH_IgnoreTrivialABI,

TAH_ConsiderTrivialABI
};

bool SpecialMemberIsTrivial(CXXMethodDecl *MD, CXXSpecialMember CSM,
TrivialABIHandling TAH = TAH_IgnoreTrivialABI,
bool Diagnose = false);
CXXSpecialMember getSpecialMember(const CXXMethodDecl *MD);
void ActOnLastBitfield(SourceLocation DeclStart,
SmallVectorImpl<Decl *> &AllIvarDecls);
Decl *ActOnIvar(Scope *S, SourceLocation DeclStart,
Declarator &D, Expr *BitfieldWidth,
tok::ObjCKeywordKind visibility);

void ActOnFields(Scope *S, SourceLocation RecLoc, Decl *TagDecl,
ArrayRef<Decl *> Fields, SourceLocation LBrac,
SourceLocation RBrac, const ParsedAttributesView &AttrList);

void ActOnTagStartDefinition(Scope *S, Decl *TagDecl);

bool ActOnDuplicateDefinition(DeclSpec &DS, Decl *Prev,
SkipBodyInfo &SkipBody);

typedef void *SkippedDefinitionContext;

SkippedDefinitionContext ActOnTagStartSkippedDefinition(Scope *S, Decl *TD);

Decl *ActOnObjCContainerStartDefinition(Decl *IDecl);

void ActOnStartCXXMemberDeclarations(Scope *S, Decl *TagDecl,
SourceLocation FinalLoc,
bool IsFinalSpelledSealed,
SourceLocation LBraceLoc);

void ActOnTagFinishDefinition(Scope *S, Decl *TagDecl,
SourceRange BraceRange);

void ActOnTagFinishSkippedDefinition(SkippedDefinitionContext Context);

void ActOnObjCContainerFinishDefinition();

void ActOnObjCTemporaryExitContainerContext(DeclContext *DC);
void ActOnObjCReenterContainerContext(DeclContext *DC);

void ActOnTagDefinitionError(Scope *S, Decl *TagDecl);

EnumConstantDecl *CheckEnumConstant(EnumDecl *Enum,
EnumConstantDecl *LastEnumConst,
SourceLocation IdLoc,
IdentifierInfo *Id,
Expr *val);
bool CheckEnumUnderlyingType(TypeSourceInfo *TI);
bool CheckEnumRedeclaration(SourceLocation EnumLoc, bool IsScoped,
QualType EnumUnderlyingTy, bool IsFixed,
const EnumDecl *Prev);

SkipBodyInfo shouldSkipAnonEnumBody(Scope *S, IdentifierInfo *II,
SourceLocation IILoc);

Decl *ActOnEnumConstant(Scope *S, Decl *EnumDecl, Decl *LastEnumConstant,
SourceLocation IdLoc, IdentifierInfo *Id,
const ParsedAttributesView &Attrs,
SourceLocation EqualLoc, Expr *Val);
void ActOnEnumBody(SourceLocation EnumLoc, SourceRange BraceRange,
Decl *EnumDecl, ArrayRef<Decl *> Elements, Scope *S,
const ParsedAttributesView &Attr);

DeclContext *getContainingDC(DeclContext *DC);

void PushDeclContext(Scope *S, DeclContext *DC);
void PopDeclContext();

void EnterDeclaratorContext(Scope *S, DeclContext *DC);
void ExitDeclaratorContext(Scope *S);

void ActOnReenterFunctionContext(Scope* S, Decl* D);
void ActOnExitFunctionContext();

DeclContext *getFunctionLevelDeclContext();

FunctionDecl *getCurFunctionDecl();

ObjCMethodDecl *getCurMethodDecl();

NamedDecl *getCurFunctionOrMethodDecl();

void PushOnScopeChains(NamedDecl *D, Scope *S, bool AddToContext = true);

void pushExternalDeclIntoScope(NamedDecl *D, DeclarationName Name);

bool isDeclInScope(NamedDecl *D, DeclContext *Ctx, Scope *S = nullptr,
bool AllowInlineNamespace = false);

static Scope *getScopeForDeclContext(Scope *S, DeclContext *DC);

TypedefDecl *ParseTypedefDecl(Scope *S, Declarator &D, QualType T,
TypeSourceInfo *TInfo);
bool isIncompatibleTypedef(TypeDecl *Old, TypedefNameDecl *New);

enum AvailabilityMergeKind {
AMK_None,
AMK_Redeclaration,
AMK_Override,
AMK_ProtocolImplementation,
};

AvailabilityAttr *mergeAvailabilityAttr(NamedDecl *D, SourceRange Range,
IdentifierInfo *Platform,
bool Implicit,
VersionTuple Introduced,
VersionTuple Deprecated,
VersionTuple Obsoleted,
bool IsUnavailable,
StringRef Message,
bool IsStrict, StringRef Replacement,
AvailabilityMergeKind AMK,
unsigned AttrSpellingListIndex);
TypeVisibilityAttr *mergeTypeVisibilityAttr(Decl *D, SourceRange Range,
TypeVisibilityAttr::VisibilityType Vis,
unsigned AttrSpellingListIndex);
VisibilityAttr *mergeVisibilityAttr(Decl *D, SourceRange Range,
VisibilityAttr::VisibilityType Vis,
unsigned AttrSpellingListIndex);
UuidAttr *mergeUuidAttr(Decl *D, SourceRange Range,
unsigned AttrSpellingListIndex, StringRef Uuid);
DLLImportAttr *mergeDLLImportAttr(Decl *D, SourceRange Range,
unsigned AttrSpellingListIndex);
DLLExportAttr *mergeDLLExportAttr(Decl *D, SourceRange Range,
unsigned AttrSpellingListIndex);
MSInheritanceAttr *
mergeMSInheritanceAttr(Decl *D, SourceRange Range, bool BestCase,
unsigned AttrSpellingListIndex,
MSInheritanceAttr::Spelling SemanticSpelling);
FormatAttr *mergeFormatAttr(Decl *D, SourceRange Range,
IdentifierInfo *Format, int FormatIdx,
int FirstArg, unsigned AttrSpellingListIndex);
SectionAttr *mergeSectionAttr(Decl *D, SourceRange Range, StringRef Name,
unsigned AttrSpellingListIndex);
CodeSegAttr *mergeCodeSegAttr(Decl *D, SourceRange Range, StringRef Name,
unsigned AttrSpellingListIndex);
AlwaysInlineAttr *mergeAlwaysInlineAttr(Decl *D, SourceRange Range,
IdentifierInfo *Ident,
unsigned AttrSpellingListIndex);
MinSizeAttr *mergeMinSizeAttr(Decl *D, SourceRange Range,
unsigned AttrSpellingListIndex);
OptimizeNoneAttr *mergeOptimizeNoneAttr(Decl *D, SourceRange Range,
unsigned AttrSpellingListIndex);
InternalLinkageAttr *mergeInternalLinkageAttr(Decl *D, SourceRange Range,
IdentifierInfo *Ident,
unsigned AttrSpellingListIndex);
CommonAttr *mergeCommonAttr(Decl *D, SourceRange Range, IdentifierInfo *Ident,
unsigned AttrSpellingListIndex);

void mergeDeclAttributes(NamedDecl *New, Decl *Old,
AvailabilityMergeKind AMK = AMK_Redeclaration);
void MergeTypedefNameDecl(Scope *S, TypedefNameDecl *New,
LookupResult &OldDecls);
bool MergeFunctionDecl(FunctionDecl *New, NamedDecl *&Old, Scope *S,
bool MergeTypeWithOld);
bool MergeCompatibleFunctionDecls(FunctionDecl *New, FunctionDecl *Old,
Scope *S, bool MergeTypeWithOld);
void mergeObjCMethodDecls(ObjCMethodDecl *New, ObjCMethodDecl *Old);
void MergeVarDecl(VarDecl *New, LookupResult &Previous);
void MergeVarDeclTypes(VarDecl *New, VarDecl *Old, bool MergeTypeWithOld);
void MergeVarDeclExceptionSpecs(VarDecl *New, VarDecl *Old);
bool checkVarDeclRedefinition(VarDecl *OldDefn, VarDecl *NewDefn);
void notePreviousDefinition(const NamedDecl *Old, SourceLocation New);
bool MergeCXXFunctionDecl(FunctionDecl *New, FunctionDecl *Old, Scope *S);

enum AssignmentAction {
AA_Assigning,
AA_Passing,
AA_Returning,
AA_Converting,
AA_Initializing,
AA_Sending,
AA_Casting,
AA_Passing_CFAudited
};

enum OverloadKind {
Ovl_Overload,

Ovl_Match,

Ovl_NonFunction
};
OverloadKind CheckOverload(Scope *S,
FunctionDecl *New,
const LookupResult &OldDecls,
NamedDecl *&OldDecl,
bool IsForUsingDecl);
bool IsOverload(FunctionDecl *New, FunctionDecl *Old, bool IsForUsingDecl,
bool ConsiderCudaAttrs = true);

bool isFunctionConsideredUnavailable(FunctionDecl *FD);

ImplicitConversionSequence
TryImplicitConversion(Expr *From, QualType ToType,
bool SuppressUserConversions,
bool AllowExplicit,
bool InOverloadResolution,
bool CStyle,
bool AllowObjCWritebackConversion);

bool IsIntegralPromotion(Expr *From, QualType FromType, QualType ToType);
bool IsFloatingPointPromotion(QualType FromType, QualType ToType);
bool IsComplexPromotion(QualType FromType, QualType ToType);
bool IsPointerConversion(Expr *From, QualType FromType, QualType ToType,
bool InOverloadResolution,
QualType& ConvertedType, bool &IncompatibleObjC);
bool isObjCPointerConversion(QualType FromType, QualType ToType,
QualType& ConvertedType, bool &IncompatibleObjC);
bool isObjCWritebackConversion(QualType FromType, QualType ToType,
QualType &ConvertedType);
bool IsBlockPointerConversion(QualType FromType, QualType ToType,
QualType& ConvertedType);
bool FunctionParamTypesAreEqual(const FunctionProtoType *OldType,
const FunctionProtoType *NewType,
unsigned *ArgPos = nullptr);
void HandleFunctionTypeMismatch(PartialDiagnostic &PDiag,
QualType FromType, QualType ToType);

void maybeExtendBlockObject(ExprResult &E);
CastKind PrepareCastToObjCObjectPointer(ExprResult &E);
bool CheckPointerConversion(Expr *From, QualType ToType,
CastKind &Kind,
CXXCastPath& BasePath,
bool IgnoreBaseAccess,
bool Diagnose = true);
bool IsMemberPointerConversion(Expr *From, QualType FromType, QualType ToType,
bool InOverloadResolution,
QualType &ConvertedType);
bool CheckMemberPointerConversion(Expr *From, QualType ToType,
CastKind &Kind,
CXXCastPath &BasePath,
bool IgnoreBaseAccess);
bool IsQualificationConversion(QualType FromType, QualType ToType,
bool CStyle, bool &ObjCLifetimeConversion);
bool IsFunctionConversion(QualType FromType, QualType ToType,
QualType &ResultTy);
bool DiagnoseMultipleUserDefinedConversion(Expr *From, QualType ToType);
bool isSameOrCompatibleFunctionType(CanQualType Param, CanQualType Arg);

ExprResult PerformMoveOrCopyInitialization(const InitializedEntity &Entity,
const VarDecl *NRVOCandidate,
QualType ResultType,
Expr *Value,
bool AllowNRVO = true);

bool CanPerformCopyInitialization(const InitializedEntity &Entity,
ExprResult Init);
ExprResult PerformCopyInitialization(const InitializedEntity &Entity,
SourceLocation EqualLoc,
ExprResult Init,
bool TopLevelOfInitList = false,
bool AllowExplicit = false);
ExprResult PerformObjectArgumentInitialization(Expr *From,
NestedNameSpecifier *Qualifier,
NamedDecl *FoundDecl,
CXXMethodDecl *Method);

void checkInitializerLifetime(const InitializedEntity &Entity, Expr *Init);

ExprResult PerformContextuallyConvertToBool(Expr *From);
ExprResult PerformContextuallyConvertToObjCPointer(Expr *From);

enum CCEKind {
CCEK_CaseValue,   
CCEK_Enumerator,  
CCEK_TemplateArg, 
CCEK_NewExpr,     
CCEK_ConstexprIf  
};
ExprResult CheckConvertedConstantExpression(Expr *From, QualType T,
llvm::APSInt &Value, CCEKind CCE);
ExprResult CheckConvertedConstantExpression(Expr *From, QualType T,
APValue &Value, CCEKind CCE);

class ContextualImplicitConverter {
public:
bool Suppress;
bool SuppressConversion;

ContextualImplicitConverter(bool Suppress = false,
bool SuppressConversion = false)
: Suppress(Suppress), SuppressConversion(SuppressConversion) {}

virtual bool match(QualType T) = 0;

virtual SemaDiagnosticBuilder
diagnoseNoMatch(Sema &S, SourceLocation Loc, QualType T) = 0;

virtual SemaDiagnosticBuilder
diagnoseIncomplete(Sema &S, SourceLocation Loc, QualType T) = 0;

virtual SemaDiagnosticBuilder diagnoseExplicitConv(
Sema &S, SourceLocation Loc, QualType T, QualType ConvTy) = 0;

virtual SemaDiagnosticBuilder
noteExplicitConv(Sema &S, CXXConversionDecl *Conv, QualType ConvTy) = 0;

virtual SemaDiagnosticBuilder
diagnoseAmbiguous(Sema &S, SourceLocation Loc, QualType T) = 0;

virtual SemaDiagnosticBuilder
noteAmbiguous(Sema &S, CXXConversionDecl *Conv, QualType ConvTy) = 0;

virtual SemaDiagnosticBuilder diagnoseConversion(
Sema &S, SourceLocation Loc, QualType T, QualType ConvTy) = 0;

virtual ~ContextualImplicitConverter() {}
};

class ICEConvertDiagnoser : public ContextualImplicitConverter {
bool AllowScopedEnumerations;

public:
ICEConvertDiagnoser(bool AllowScopedEnumerations,
bool Suppress, bool SuppressConversion)
: ContextualImplicitConverter(Suppress, SuppressConversion),
AllowScopedEnumerations(AllowScopedEnumerations) {}

bool match(QualType T) override;

SemaDiagnosticBuilder
diagnoseNoMatch(Sema &S, SourceLocation Loc, QualType T) override {
return diagnoseNotInt(S, Loc, T);
}

virtual SemaDiagnosticBuilder
diagnoseNotInt(Sema &S, SourceLocation Loc, QualType T) = 0;
};

ExprResult PerformContextualImplicitConversion(
SourceLocation Loc, Expr *FromE, ContextualImplicitConverter &Converter);


enum ObjCSubscriptKind {
OS_Array,
OS_Dictionary,
OS_Error
};
ObjCSubscriptKind CheckSubscriptingKind(Expr *FromE);

enum ObjCLiteralKind {
LK_Array,
LK_Dictionary,
LK_Numeric,
LK_Boxed,
LK_String,
LK_Block,
LK_None
};
ObjCLiteralKind CheckLiteralKind(Expr *FromE);

ExprResult PerformObjectMemberConversion(Expr *From,
NestedNameSpecifier *Qualifier,
NamedDecl *FoundDecl,
NamedDecl *Member);

typedef llvm::SmallSetVector<DeclContext   *, 16> AssociatedNamespaceSet;
typedef llvm::SmallSetVector<CXXRecordDecl *, 16> AssociatedClassSet;

void AddOverloadCandidate(FunctionDecl *Function,
DeclAccessPair FoundDecl,
ArrayRef<Expr *> Args,
OverloadCandidateSet &CandidateSet,
bool SuppressUserConversions = false,
bool PartialOverloading = false,
bool AllowExplicit = false,
ConversionSequenceList EarlyConversions = None);
void AddFunctionCandidates(const UnresolvedSetImpl &Functions,
ArrayRef<Expr *> Args,
OverloadCandidateSet &CandidateSet,
TemplateArgumentListInfo *ExplicitTemplateArgs = nullptr,
bool SuppressUserConversions = false,
bool PartialOverloading = false,
bool FirstArgumentIsBase = false);
void AddMethodCandidate(DeclAccessPair FoundDecl,
QualType ObjectType,
Expr::Classification ObjectClassification,
ArrayRef<Expr *> Args,
OverloadCandidateSet& CandidateSet,
bool SuppressUserConversion = false);
void AddMethodCandidate(CXXMethodDecl *Method,
DeclAccessPair FoundDecl,
CXXRecordDecl *ActingContext, QualType ObjectType,
Expr::Classification ObjectClassification,
ArrayRef<Expr *> Args,
OverloadCandidateSet& CandidateSet,
bool SuppressUserConversions = false,
bool PartialOverloading = false,
ConversionSequenceList EarlyConversions = None);
void AddMethodTemplateCandidate(FunctionTemplateDecl *MethodTmpl,
DeclAccessPair FoundDecl,
CXXRecordDecl *ActingContext,
TemplateArgumentListInfo *ExplicitTemplateArgs,
QualType ObjectType,
Expr::Classification ObjectClassification,
ArrayRef<Expr *> Args,
OverloadCandidateSet& CandidateSet,
bool SuppressUserConversions = false,
bool PartialOverloading = false);
void AddTemplateOverloadCandidate(FunctionTemplateDecl *FunctionTemplate,
DeclAccessPair FoundDecl,
TemplateArgumentListInfo *ExplicitTemplateArgs,
ArrayRef<Expr *> Args,
OverloadCandidateSet& CandidateSet,
bool SuppressUserConversions = false,
bool PartialOverloading = false);
bool CheckNonDependentConversions(FunctionTemplateDecl *FunctionTemplate,
ArrayRef<QualType> ParamTypes,
ArrayRef<Expr *> Args,
OverloadCandidateSet &CandidateSet,
ConversionSequenceList &Conversions,
bool SuppressUserConversions,
CXXRecordDecl *ActingContext = nullptr,
QualType ObjectType = QualType(),
Expr::Classification
ObjectClassification = {});
void AddConversionCandidate(CXXConversionDecl *Conversion,
DeclAccessPair FoundDecl,
CXXRecordDecl *ActingContext,
Expr *From, QualType ToType,
OverloadCandidateSet& CandidateSet,
bool AllowObjCConversionOnExplicit,
bool AllowResultConversion = true);
void AddTemplateConversionCandidate(FunctionTemplateDecl *FunctionTemplate,
DeclAccessPair FoundDecl,
CXXRecordDecl *ActingContext,
Expr *From, QualType ToType,
OverloadCandidateSet &CandidateSet,
bool AllowObjCConversionOnExplicit,
bool AllowResultConversion = true);
void AddSurrogateCandidate(CXXConversionDecl *Conversion,
DeclAccessPair FoundDecl,
CXXRecordDecl *ActingContext,
const FunctionProtoType *Proto,
Expr *Object, ArrayRef<Expr *> Args,
OverloadCandidateSet& CandidateSet);
void AddMemberOperatorCandidates(OverloadedOperatorKind Op,
SourceLocation OpLoc, ArrayRef<Expr *> Args,
OverloadCandidateSet& CandidateSet,
SourceRange OpRange = SourceRange());
void AddBuiltinCandidate(QualType *ParamTys, ArrayRef<Expr *> Args,
OverloadCandidateSet& CandidateSet,
bool IsAssignmentOperator = false,
unsigned NumContextualBoolArguments = 0);
void AddBuiltinOperatorCandidates(OverloadedOperatorKind Op,
SourceLocation OpLoc, ArrayRef<Expr *> Args,
OverloadCandidateSet& CandidateSet);
void AddArgumentDependentLookupCandidates(DeclarationName Name,
SourceLocation Loc,
ArrayRef<Expr *> Args,
TemplateArgumentListInfo *ExplicitTemplateArgs,
OverloadCandidateSet& CandidateSet,
bool PartialOverloading = false);

void NoteOverloadCandidate(NamedDecl *Found, FunctionDecl *Fn,
QualType DestType = QualType(),
bool TakingAddress = false);

void NoteAllOverloadCandidates(Expr *E, QualType DestType = QualType(),
bool TakingAddress = false);

EnableIfAttr *CheckEnableIf(FunctionDecl *Function, ArrayRef<Expr *> Args,
bool MissingImplicitThis = false);

std::pair<Expr *, std::string>
findFailedBooleanCondition(Expr *Cond, bool AllowTopLevelCond);

bool diagnoseArgDependentDiagnoseIfAttrs(const FunctionDecl *Function,
const Expr *ThisArg,
ArrayRef<const Expr *> Args,
SourceLocation Loc);

bool diagnoseArgIndependentDiagnoseIfAttrs(const NamedDecl *ND,
SourceLocation Loc);

bool checkAddressOfFunctionIsAvailable(const FunctionDecl *Function,
bool Complain = false,
SourceLocation Loc = SourceLocation());

QualType ExtractUnqualifiedFunctionType(QualType PossiblyAFunctionType);

FunctionDecl *
ResolveAddressOfOverloadedFunction(Expr *AddressOfExpr,
QualType TargetType,
bool Complain,
DeclAccessPair &Found,
bool *pHadMultipleCandidates = nullptr);

FunctionDecl *
resolveAddressOfOnlyViableOverloadCandidate(Expr *E,
DeclAccessPair &FoundResult);

bool resolveAndFixAddressOfOnlyViableOverloadCandidate(
ExprResult &SrcExpr, bool DoFunctionPointerConversion = false);

FunctionDecl *
ResolveSingleFunctionTemplateSpecialization(OverloadExpr *ovl,
bool Complain = false,
DeclAccessPair *Found = nullptr);

bool ResolveAndFixSingleFunctionTemplateSpecialization(
ExprResult &SrcExpr,
bool DoFunctionPointerConverion = false,
bool Complain = false,
SourceRange OpRangeForComplaining = SourceRange(),
QualType DestTypeForComplaining = QualType(),
unsigned DiagIDForComplaining = 0);


Expr *FixOverloadedFunctionReference(Expr *E,
DeclAccessPair FoundDecl,
FunctionDecl *Fn);
ExprResult FixOverloadedFunctionReference(ExprResult,
DeclAccessPair FoundDecl,
FunctionDecl *Fn);

void AddOverloadedCallCandidates(UnresolvedLookupExpr *ULE,
ArrayRef<Expr *> Args,
OverloadCandidateSet &CandidateSet,
bool PartialOverloading = false);

enum ForRangeStatus {
FRS_Success,
FRS_NoViableFunction,
FRS_DiagnosticIssued
};

ForRangeStatus BuildForRangeBeginEndCall(SourceLocation Loc,
SourceLocation RangeLoc,
const DeclarationNameInfo &NameInfo,
LookupResult &MemberLookup,
OverloadCandidateSet *CandidateSet,
Expr *Range, ExprResult *CallExpr);

ExprResult BuildOverloadedCallExpr(Scope *S, Expr *Fn,
UnresolvedLookupExpr *ULE,
SourceLocation LParenLoc,
MultiExprArg Args,
SourceLocation RParenLoc,
Expr *ExecConfig,
bool AllowTypoCorrection=true,
bool CalleesAddressIsTaken=false);

bool buildOverloadedCallSet(Scope *S, Expr *Fn, UnresolvedLookupExpr *ULE,
MultiExprArg Args, SourceLocation RParenLoc,
OverloadCandidateSet *CandidateSet,
ExprResult *Result);

ExprResult CreateOverloadedUnaryOp(SourceLocation OpLoc,
UnaryOperatorKind Opc,
const UnresolvedSetImpl &Fns,
Expr *input, bool RequiresADL = true);

ExprResult CreateOverloadedBinOp(SourceLocation OpLoc,
BinaryOperatorKind Opc,
const UnresolvedSetImpl &Fns,
Expr *LHS, Expr *RHS,
bool RequiresADL = true);

ExprResult CreateOverloadedArraySubscriptExpr(SourceLocation LLoc,
SourceLocation RLoc,
Expr *Base,Expr *Idx);

ExprResult
BuildCallToMemberFunction(Scope *S, Expr *MemExpr,
SourceLocation LParenLoc,
MultiExprArg Args,
SourceLocation RParenLoc);
ExprResult
BuildCallToObjectOfClassType(Scope *S, Expr *Object, SourceLocation LParenLoc,
MultiExprArg Args,
SourceLocation RParenLoc);

ExprResult BuildOverloadedArrowExpr(Scope *S, Expr *Base,
SourceLocation OpLoc,
bool *NoArrowOperatorFound = nullptr);

bool CheckCallReturnType(QualType ReturnType, SourceLocation Loc,
CallExpr *CE, FunctionDecl *FD);

bool CheckParmsForFunctionDef(ArrayRef<ParmVarDecl *> Parameters,
bool CheckParameterNames);
void CheckCXXDefaultArguments(FunctionDecl *FD);
void CheckExtraCXXDefaultArguments(Declarator &D);
Scope *getNonFieldDeclScope(Scope *S);


enum LookupNameKind {
LookupOrdinaryName = 0,
LookupTagName,
LookupLabel,
LookupMemberName,
LookupOperatorName,
LookupNestedNameSpecifierName,
LookupNamespaceName,
LookupUsingDeclName,
LookupRedeclarationWithLinkage,
LookupLocalFriendName,
LookupObjCProtocolName,
LookupObjCImplicitSelfParam,
LookupOMPReductionName,
LookupAnyName
};

enum RedeclarationKind {
NotForRedeclaration = 0,
ForVisibleRedeclaration,
ForExternalRedeclaration
};

RedeclarationKind forRedeclarationInCurContext() {
if (cast<Decl>(CurContext)
->getOwningModuleForLinkage(true))
return ForVisibleRedeclaration;
return ForExternalRedeclaration;
}

enum LiteralOperatorLookupResult {
LOLR_Error,
LOLR_ErrorNoDiagnostic,
LOLR_Cooked,
LOLR_Raw,
LOLR_Template,
LOLR_StringTemplate
};

SpecialMemberOverloadResult LookupSpecialMember(CXXRecordDecl *D,
CXXSpecialMember SM,
bool ConstArg,
bool VolatileArg,
bool RValueThis,
bool ConstThis,
bool VolatileThis);

typedef std::function<void(const TypoCorrection &)> TypoDiagnosticGenerator;
typedef std::function<ExprResult(Sema &, TypoExpr *, TypoCorrection)>
TypoRecoveryCallback;

private:
bool CppLookupName(LookupResult &R, Scope *S);

struct TypoExprState {
std::unique_ptr<TypoCorrectionConsumer> Consumer;
TypoDiagnosticGenerator DiagHandler;
TypoRecoveryCallback RecoveryHandler;
TypoExprState();
TypoExprState(TypoExprState &&other) noexcept;
TypoExprState &operator=(TypoExprState &&other) noexcept;
};

llvm::MapVector<TypoExpr *, TypoExprState> DelayedTypos;

TypoExpr *createDelayedTypo(std::unique_ptr<TypoCorrectionConsumer> TCC,
TypoDiagnosticGenerator TDG,
TypoRecoveryCallback TRC);

llvm::MapVector<NamespaceDecl*, bool> KnownNamespaces;

bool LoadedExternalKnownNamespaces;

std::unique_ptr<TypoCorrectionConsumer>
makeTypoCorrectionConsumer(const DeclarationNameInfo &Typo,
Sema::LookupNameKind LookupKind, Scope *S,
CXXScopeSpec *SS,
std::unique_ptr<CorrectionCandidateCallback> CCC,
DeclContext *MemberContext, bool EnteringContext,
const ObjCObjectPointerType *OPT,
bool ErrorRecovery);

public:
const TypoExprState &getTypoExprState(TypoExpr *TE) const;

void clearDelayedTypo(TypoExpr *TE);

NamedDecl *LookupSingleName(Scope *S, DeclarationName Name,
SourceLocation Loc,
LookupNameKind NameKind,
RedeclarationKind Redecl
= NotForRedeclaration);
bool LookupName(LookupResult &R, Scope *S,
bool AllowBuiltinCreation = false);
bool LookupQualifiedName(LookupResult &R, DeclContext *LookupCtx,
bool InUnqualifiedLookup = false);
bool LookupQualifiedName(LookupResult &R, DeclContext *LookupCtx,
CXXScopeSpec &SS);
bool LookupParsedName(LookupResult &R, Scope *S, CXXScopeSpec *SS,
bool AllowBuiltinCreation = false,
bool EnteringContext = false);
ObjCProtocolDecl *LookupProtocol(IdentifierInfo *II, SourceLocation IdLoc,
RedeclarationKind Redecl
= NotForRedeclaration);
bool LookupInSuper(LookupResult &R, CXXRecordDecl *Class);

void LookupOverloadedOperatorName(OverloadedOperatorKind Op, Scope *S,
QualType T1, QualType T2,
UnresolvedSetImpl &Functions);

LabelDecl *LookupOrCreateLabel(IdentifierInfo *II, SourceLocation IdentLoc,
SourceLocation GnuLabelLoc = SourceLocation());

DeclContextLookupResult LookupConstructors(CXXRecordDecl *Class);
CXXConstructorDecl *LookupDefaultConstructor(CXXRecordDecl *Class);
CXXConstructorDecl *LookupCopyingConstructor(CXXRecordDecl *Class,
unsigned Quals);
CXXMethodDecl *LookupCopyingAssignment(CXXRecordDecl *Class, unsigned Quals,
bool RValueThis, unsigned ThisQuals);
CXXConstructorDecl *LookupMovingConstructor(CXXRecordDecl *Class,
unsigned Quals);
CXXMethodDecl *LookupMovingAssignment(CXXRecordDecl *Class, unsigned Quals,
bool RValueThis, unsigned ThisQuals);
CXXDestructorDecl *LookupDestructor(CXXRecordDecl *Class);

bool checkLiteralOperatorId(const CXXScopeSpec &SS, const UnqualifiedId &Id);
LiteralOperatorLookupResult LookupLiteralOperator(Scope *S, LookupResult &R,
ArrayRef<QualType> ArgTys,
bool AllowRaw,
bool AllowTemplate,
bool AllowStringTemplate,
bool DiagnoseMissing);
bool isKnownName(StringRef name);

void ArgumentDependentLookup(DeclarationName Name, SourceLocation Loc,
ArrayRef<Expr *> Args, ADLResult &Functions);

void LookupVisibleDecls(Scope *S, LookupNameKind Kind,
VisibleDeclConsumer &Consumer,
bool IncludeGlobalScope = true,
bool LoadExternal = true);
void LookupVisibleDecls(DeclContext *Ctx, LookupNameKind Kind,
VisibleDeclConsumer &Consumer,
bool IncludeGlobalScope = true,
bool IncludeDependentBases = false,
bool LoadExternal = true);

enum CorrectTypoKind {
CTK_NonError,     
CTK_ErrorRecovery 
};

TypoCorrection CorrectTypo(const DeclarationNameInfo &Typo,
Sema::LookupNameKind LookupKind,
Scope *S, CXXScopeSpec *SS,
std::unique_ptr<CorrectionCandidateCallback> CCC,
CorrectTypoKind Mode,
DeclContext *MemberContext = nullptr,
bool EnteringContext = false,
const ObjCObjectPointerType *OPT = nullptr,
bool RecordFailure = true);

TypoExpr *CorrectTypoDelayed(const DeclarationNameInfo &Typo,
Sema::LookupNameKind LookupKind, Scope *S,
CXXScopeSpec *SS,
std::unique_ptr<CorrectionCandidateCallback> CCC,
TypoDiagnosticGenerator TDG,
TypoRecoveryCallback TRC, CorrectTypoKind Mode,
DeclContext *MemberContext = nullptr,
bool EnteringContext = false,
const ObjCObjectPointerType *OPT = nullptr);

ExprResult
CorrectDelayedTyposInExpr(Expr *E, VarDecl *InitDecl = nullptr,
llvm::function_ref<ExprResult(Expr *)> Filter =
[](Expr *E) -> ExprResult { return E; });

ExprResult
CorrectDelayedTyposInExpr(Expr *E,
llvm::function_ref<ExprResult(Expr *)> Filter) {
return CorrectDelayedTyposInExpr(E, nullptr, Filter);
}

ExprResult
CorrectDelayedTyposInExpr(ExprResult ER, VarDecl *InitDecl = nullptr,
llvm::function_ref<ExprResult(Expr *)> Filter =
[](Expr *E) -> ExprResult { return E; }) {
return ER.isInvalid() ? ER : CorrectDelayedTyposInExpr(ER.get(), Filter);
}

ExprResult
CorrectDelayedTyposInExpr(ExprResult ER,
llvm::function_ref<ExprResult(Expr *)> Filter) {
return CorrectDelayedTyposInExpr(ER, nullptr, Filter);
}

void diagnoseTypo(const TypoCorrection &Correction,
const PartialDiagnostic &TypoDiag,
bool ErrorRecovery = true);

void diagnoseTypo(const TypoCorrection &Correction,
const PartialDiagnostic &TypoDiag,
const PartialDiagnostic &PrevNote,
bool ErrorRecovery = true);

void MarkTypoCorrectedFunctionDefinition(const NamedDecl *F);

void FindAssociatedClassesAndNamespaces(SourceLocation InstantiationLoc,
ArrayRef<Expr *> Args,
AssociatedNamespaceSet &AssociatedNamespaces,
AssociatedClassSet &AssociatedClasses);

void FilterLookupForScope(LookupResult &R, DeclContext *Ctx, Scope *S,
bool ConsiderLinkage, bool AllowInlineNamespace);

bool CheckRedeclarationModuleOwnership(NamedDecl *New, NamedDecl *Old);

void DiagnoseAmbiguousLookup(LookupResult &Result);

ObjCInterfaceDecl *getObjCInterfaceDecl(IdentifierInfo *&Id,
SourceLocation IdLoc,
bool TypoCorrection = false);
NamedDecl *LazilyCreateBuiltin(IdentifierInfo *II, unsigned ID,
Scope *S, bool ForRedeclaration,
SourceLocation Loc);
NamedDecl *ImplicitlyDefineFunction(SourceLocation Loc, IdentifierInfo &II,
Scope *S);
void AddKnownFunctionAttributes(FunctionDecl *FD);


void ProcessPragmaWeak(Scope *S, Decl *D);
void ProcessDeclAttributes(Scope *S, Decl *D, const Declarator &PD);
void ProcessDeclAttributeDelayed(Decl *D,
const ParsedAttributesView &AttrList);
void ProcessDeclAttributeList(Scope *S, Decl *D, const ParsedAttributesView &AL,
bool IncludeCXX11Attributes = true);
bool ProcessAccessDeclAttributeList(AccessSpecDecl *ASDecl,
const ParsedAttributesView &AttrList);

void checkUnusedDeclAttributes(Declarator &D);

bool isValidPointerAttrType(QualType T, bool RefOkay = false);

bool CheckRegparmAttr(const ParsedAttr &attr, unsigned &value);
bool CheckCallingConvAttr(const ParsedAttr &attr, CallingConv &CC,
const FunctionDecl *FD = nullptr);
bool CheckAttrTarget(const ParsedAttr &CurrAttr);
bool CheckAttrNoArgs(const ParsedAttr &CurrAttr);
bool checkStringLiteralArgumentAttr(const ParsedAttr &Attr, unsigned ArgNum,
StringRef &Str,
SourceLocation *ArgLocation = nullptr);
bool checkSectionName(SourceLocation LiteralLoc, StringRef Str);
bool checkTargetAttr(SourceLocation LiteralLoc, StringRef Str);
bool checkMSInheritanceAttrOnDefinition(
CXXRecordDecl *RD, SourceRange Range, bool BestCase,
MSInheritanceAttr::Spelling SemanticSpelling);

void CheckAlignasUnderalignment(Decl *D);

void adjustMemberFunctionCC(QualType &T, bool IsStatic, bool IsCtorOrDtor,
SourceLocation Loc);

bool hasExplicitCallingConv(QualType &T);

const AttributedType *getCallingConvAttributedType(QualType T) const;

bool checkNullabilityTypeSpecifier(QualType &type, NullabilityKind nullability,
SourceLocation nullabilityLoc,
bool isContextSensitive,
bool allowArrayTypes);

StmtResult ProcessStmtAttributes(Stmt *Stmt,
const ParsedAttributesView &Attrs,
SourceRange Range);

void WarnConflictingTypedMethods(ObjCMethodDecl *Method,
ObjCMethodDecl *MethodDecl,
bool IsProtocolMethodDecl);

void CheckConflictingOverridingMethod(ObjCMethodDecl *Method,
ObjCMethodDecl *Overridden,
bool IsProtocolMethodDecl);

void WarnExactTypedMethods(ObjCMethodDecl *Method,
ObjCMethodDecl *MethodDecl,
bool IsProtocolMethodDecl);

typedef llvm::SmallPtrSet<Selector, 8> SelectorSet;

void CheckImplementationIvars(ObjCImplementationDecl *ImpDecl,
ObjCIvarDecl **Fields, unsigned nIvars,
SourceLocation Loc);

void ImplMethodsVsClassMethods(Scope *S, ObjCImplDecl* IMPDecl,
ObjCContainerDecl* IDecl,
bool IncompleteImpl = false);

void DiagnoseUnimplementedProperties(Scope *S, ObjCImplDecl* IMPDecl,
ObjCContainerDecl *CDecl,
bool SynthesizeProperties);

void diagnoseNullResettableSynthesizedSetters(const ObjCImplDecl *impDecl);

void DefaultSynthesizeProperties(Scope *S, ObjCImplDecl *IMPDecl,
ObjCInterfaceDecl *IDecl,
SourceLocation AtEnd);
void DefaultSynthesizeProperties(Scope *S, Decl *D, SourceLocation AtEnd);

bool IvarBacksCurrentMethodAccessor(ObjCInterfaceDecl *IFace,
ObjCMethodDecl *Method, ObjCIvarDecl *IV);

void DiagnoseUnusedBackingIvarInAccessor(Scope *S,
const ObjCImplementationDecl *ImplD);

ObjCIvarDecl *GetIvarBackingPropertyAccessor(const ObjCMethodDecl *Method,
const ObjCPropertyDecl *&PDecl) const;

ObjCPropertyDecl *HandlePropertyInClassExtension(Scope *S,
SourceLocation AtLoc,
SourceLocation LParenLoc,
FieldDeclarator &FD,
Selector GetterSel,
SourceLocation GetterNameLoc,
Selector SetterSel,
SourceLocation SetterNameLoc,
const bool isReadWrite,
unsigned &Attributes,
const unsigned AttributesAsWritten,
QualType T,
TypeSourceInfo *TSI,
tok::ObjCKeywordKind MethodImplKind);

ObjCPropertyDecl *CreatePropertyDecl(Scope *S,
ObjCContainerDecl *CDecl,
SourceLocation AtLoc,
SourceLocation LParenLoc,
FieldDeclarator &FD,
Selector GetterSel,
SourceLocation GetterNameLoc,
Selector SetterSel,
SourceLocation SetterNameLoc,
const bool isReadWrite,
const unsigned Attributes,
const unsigned AttributesAsWritten,
QualType T,
TypeSourceInfo *TSI,
tok::ObjCKeywordKind MethodImplKind,
DeclContext *lexicalDC = nullptr);

void AtomicPropertySetterGetterRules(ObjCImplDecl* IMPDecl,
ObjCInterfaceDecl* IDecl);

void DiagnoseOwningPropertyGetterSynthesis(const ObjCImplementationDecl *D);

void DiagnoseMissingDesignatedInitOverrides(
const ObjCImplementationDecl *ImplD,
const ObjCInterfaceDecl *IFD);

void DiagnoseDuplicateIvars(ObjCInterfaceDecl *ID, ObjCInterfaceDecl *SID);

enum MethodMatchStrategy {
MMS_loose,
MMS_strict
};

bool MatchTwoMethodDeclarations(const ObjCMethodDecl *Method,
const ObjCMethodDecl *PrevMethod,
MethodMatchStrategy strategy = MMS_strict);

void MatchAllMethodDeclarations(const SelectorSet &InsMap,
const SelectorSet &ClsMap,
SelectorSet &InsMapSeen,
SelectorSet &ClsMapSeen,
ObjCImplDecl* IMPDecl,
ObjCContainerDecl* IDecl,
bool &IncompleteImpl,
bool ImmediateClass,
bool WarnCategoryMethodImpl=false);

void CheckCategoryVsClassMethodMatches(ObjCCategoryImplDecl *CatIMP);

void addMethodToGlobalList(ObjCMethodList *List, ObjCMethodDecl *Method);

private:
void AddMethodToGlobalPool(ObjCMethodDecl *Method, bool impl, bool instance);

ObjCMethodDecl *LookupMethodInGlobalPool(Selector Sel, SourceRange R,
bool receiverIdOrClass,
bool instance);

public:
bool
CollectMultipleMethodsInGlobalPool(Selector Sel,
SmallVectorImpl<ObjCMethodDecl*>& Methods,
bool InstanceFirst, bool CheckTheOther,
const ObjCObjectType *TypeBound = nullptr);

bool
AreMultipleMethodsInGlobalPool(Selector Sel, ObjCMethodDecl *BestMethod,
SourceRange R, bool receiverIdOrClass,
SmallVectorImpl<ObjCMethodDecl*>& Methods);

void
DiagnoseMultipleMethodInGlobalPool(SmallVectorImpl<ObjCMethodDecl*> &Methods,
Selector Sel, SourceRange R,
bool receiverIdOrClass);

private:
ObjCMethodDecl *SelectBestMethod(Selector Sel, MultiExprArg Args,
bool IsInstance,
SmallVectorImpl<ObjCMethodDecl*>& Methods);


TypoCorrection FailedCorrection(IdentifierInfo *Typo, SourceLocation TypoLoc,
bool RecordFailure = true) {
if (RecordFailure)
TypoCorrectionFailures[Typo].insert(TypoLoc);
return TypoCorrection();
}

public:
void AddInstanceMethodToGlobalPool(ObjCMethodDecl *Method, bool impl=false) {
AddMethodToGlobalPool(Method, impl, true);
}

void AddFactoryMethodToGlobalPool(ObjCMethodDecl *Method, bool impl=false) {
AddMethodToGlobalPool(Method, impl, false);
}

void AddAnyMethodToGlobalPool(Decl *D);

ObjCMethodDecl *LookupInstanceMethodInGlobalPool(Selector Sel, SourceRange R,
bool receiverIdOrClass=false) {
return LookupMethodInGlobalPool(Sel, R, receiverIdOrClass,
true);
}

ObjCMethodDecl *LookupFactoryMethodInGlobalPool(Selector Sel, SourceRange R,
bool receiverIdOrClass=false) {
return LookupMethodInGlobalPool(Sel, R, receiverIdOrClass,
false);
}

const ObjCMethodDecl *SelectorsForTypoCorrection(Selector Sel,
QualType ObjectType=QualType());
ObjCMethodDecl *LookupImplementedMethodInGlobalPool(Selector Sel);

void CollectIvarsToConstructOrDestruct(ObjCInterfaceDecl *OI,
SmallVectorImpl<ObjCIvarDecl*> &Ivars);

public:
class FullExprArg {
public:
FullExprArg() : E(nullptr) { }
FullExprArg(Sema &actions) : E(nullptr) { }

ExprResult release() {
return E;
}

Expr *get() const { return E; }

Expr *operator->() {
return E;
}

private:
friend class Sema;

explicit FullExprArg(Expr *expr) : E(expr) {}

Expr *E;
};

FullExprArg MakeFullExpr(Expr *Arg) {
return MakeFullExpr(Arg, Arg ? Arg->getExprLoc() : SourceLocation());
}
FullExprArg MakeFullExpr(Expr *Arg, SourceLocation CC) {
return FullExprArg(ActOnFinishFullExpr(Arg, CC).get());
}
FullExprArg MakeFullDiscardedValueExpr(Expr *Arg) {
ExprResult FE =
ActOnFinishFullExpr(Arg, Arg ? Arg->getExprLoc() : SourceLocation(),
true);
return FullExprArg(FE.get());
}

StmtResult ActOnExprStmt(ExprResult Arg);
StmtResult ActOnExprStmtError();

StmtResult ActOnNullStmt(SourceLocation SemiLoc,
bool HasLeadingEmptyMacro = false);

void ActOnStartOfCompoundStmt(bool IsStmtExpr);
void ActOnFinishOfCompoundStmt();
StmtResult ActOnCompoundStmt(SourceLocation L, SourceLocation R,
ArrayRef<Stmt *> Elts, bool isStmtExpr);

class CompoundScopeRAII {
public:
CompoundScopeRAII(Sema &S, bool IsStmtExpr = false) : S(S) {
S.ActOnStartOfCompoundStmt(IsStmtExpr);
}

~CompoundScopeRAII() {
S.ActOnFinishOfCompoundStmt();
}

private:
Sema &S;
};

struct FunctionScopeRAII {
Sema &S;
bool Active;
FunctionScopeRAII(Sema &S) : S(S), Active(true) {}
~FunctionScopeRAII() {
if (Active)
S.PopFunctionScopeInfo();
}
void disable() { Active = false; }
};

StmtResult ActOnDeclStmt(DeclGroupPtrTy Decl,
SourceLocation StartLoc,
SourceLocation EndLoc);
void ActOnForEachDeclStmt(DeclGroupPtrTy Decl);
StmtResult ActOnForEachLValueExpr(Expr *E);
ExprResult ActOnCaseExpr(SourceLocation CaseLoc, ExprResult Val);
StmtResult ActOnCaseStmt(SourceLocation CaseLoc, ExprResult LHS,
SourceLocation DotDotDotLoc, ExprResult RHS,
SourceLocation ColonLoc);
void ActOnCaseStmtBody(Stmt *CaseStmt, Stmt *SubStmt);

StmtResult ActOnDefaultStmt(SourceLocation DefaultLoc,
SourceLocation ColonLoc,
Stmt *SubStmt, Scope *CurScope);
StmtResult ActOnLabelStmt(SourceLocation IdentLoc, LabelDecl *TheDecl,
SourceLocation ColonLoc, Stmt *SubStmt);

StmtResult ActOnAttributedStmt(SourceLocation AttrLoc,
ArrayRef<const Attr*> Attrs,
Stmt *SubStmt);

class ConditionResult;
StmtResult ActOnIfStmt(SourceLocation IfLoc, bool IsConstexpr,
Stmt *InitStmt,
ConditionResult Cond, Stmt *ThenVal,
SourceLocation ElseLoc, Stmt *ElseVal);
StmtResult BuildIfStmt(SourceLocation IfLoc, bool IsConstexpr,
Stmt *InitStmt,
ConditionResult Cond, Stmt *ThenVal,
SourceLocation ElseLoc, Stmt *ElseVal);
StmtResult ActOnStartOfSwitchStmt(SourceLocation SwitchLoc,
Stmt *InitStmt,
ConditionResult Cond);
StmtResult ActOnFinishSwitchStmt(SourceLocation SwitchLoc,
Stmt *Switch, Stmt *Body);
StmtResult ActOnWhileStmt(SourceLocation WhileLoc, ConditionResult Cond,
Stmt *Body);
StmtResult ActOnDoStmt(SourceLocation DoLoc, Stmt *Body,
SourceLocation WhileLoc, SourceLocation CondLParen,
Expr *Cond, SourceLocation CondRParen);

StmtResult ActOnForStmt(SourceLocation ForLoc,
SourceLocation LParenLoc,
Stmt *First,
ConditionResult Second,
FullExprArg Third,
SourceLocation RParenLoc,
Stmt *Body);
ExprResult CheckObjCForCollectionOperand(SourceLocation forLoc,
Expr *collection);
StmtResult ActOnObjCForCollectionStmt(SourceLocation ForColLoc,
Stmt *First, Expr *collection,
SourceLocation RParenLoc);
StmtResult FinishObjCForCollectionStmt(Stmt *ForCollection, Stmt *Body);

enum BuildForRangeKind {
BFRK_Build,
BFRK_Rebuild,
BFRK_Check
};

StmtResult ActOnCXXForRangeStmt(Scope *S, SourceLocation ForLoc,
SourceLocation CoawaitLoc,
Stmt *LoopVar,
SourceLocation ColonLoc, Expr *Collection,
SourceLocation RParenLoc,
BuildForRangeKind Kind);
StmtResult BuildCXXForRangeStmt(SourceLocation ForLoc,
SourceLocation CoawaitLoc,
SourceLocation ColonLoc,
Stmt *RangeDecl, Stmt *Begin, Stmt *End,
Expr *Cond, Expr *Inc,
Stmt *LoopVarDecl,
SourceLocation RParenLoc,
BuildForRangeKind Kind);
StmtResult FinishCXXForRangeStmt(Stmt *ForRange, Stmt *Body);

StmtResult ActOnGotoStmt(SourceLocation GotoLoc,
SourceLocation LabelLoc,
LabelDecl *TheDecl);
StmtResult ActOnIndirectGotoStmt(SourceLocation GotoLoc,
SourceLocation StarLoc,
Expr *DestExp);
StmtResult ActOnContinueStmt(SourceLocation ContinueLoc, Scope *CurScope);
StmtResult ActOnBreakStmt(SourceLocation BreakLoc, Scope *CurScope);

void ActOnCapturedRegionStart(SourceLocation Loc, Scope *CurScope,
CapturedRegionKind Kind, unsigned NumParams);
typedef std::pair<StringRef, QualType> CapturedParamNameType;
void ActOnCapturedRegionStart(SourceLocation Loc, Scope *CurScope,
CapturedRegionKind Kind,
ArrayRef<CapturedParamNameType> Params);
StmtResult ActOnCapturedRegionEnd(Stmt *S);
void ActOnCapturedRegionError();
RecordDecl *CreateCapturedStmtRecordDecl(CapturedDecl *&CD,
SourceLocation Loc,
unsigned NumParams);

enum CopyElisionSemanticsKind {
CES_Strict = 0,
CES_AllowParameters = 1,
CES_AllowDifferentTypes = 2,
CES_AllowExceptionVariables = 4,
CES_FormerDefault = (CES_AllowParameters),
CES_Default = (CES_AllowParameters | CES_AllowDifferentTypes),
CES_AsIfByStdMove = (CES_AllowParameters | CES_AllowDifferentTypes |
CES_AllowExceptionVariables),
};

VarDecl *getCopyElisionCandidate(QualType ReturnType, Expr *E,
CopyElisionSemanticsKind CESK);
bool isCopyElisionCandidate(QualType ReturnType, const VarDecl *VD,
CopyElisionSemanticsKind CESK);

StmtResult ActOnReturnStmt(SourceLocation ReturnLoc, Expr *RetValExp,
Scope *CurScope);
StmtResult BuildReturnStmt(SourceLocation ReturnLoc, Expr *RetValExp);
StmtResult ActOnCapScopeReturnStmt(SourceLocation ReturnLoc, Expr *RetValExp);

StmtResult ActOnGCCAsmStmt(SourceLocation AsmLoc, bool IsSimple,
bool IsVolatile, unsigned NumOutputs,
unsigned NumInputs, IdentifierInfo **Names,
MultiExprArg Constraints, MultiExprArg Exprs,
Expr *AsmString, MultiExprArg Clobbers,
SourceLocation RParenLoc);

void FillInlineAsmIdentifierInfo(Expr *Res,
llvm::InlineAsmIdentifierInfo &Info);
ExprResult LookupInlineAsmIdentifier(CXXScopeSpec &SS,
SourceLocation TemplateKWLoc,
UnqualifiedId &Id,
bool IsUnevaluatedContext);
bool LookupInlineAsmField(StringRef Base, StringRef Member,
unsigned &Offset, SourceLocation AsmLoc);
ExprResult LookupInlineAsmVarDeclField(Expr *RefExpr, StringRef Member,
SourceLocation AsmLoc);
StmtResult ActOnMSAsmStmt(SourceLocation AsmLoc, SourceLocation LBraceLoc,
ArrayRef<Token> AsmToks,
StringRef AsmString,
unsigned NumOutputs, unsigned NumInputs,
ArrayRef<StringRef> Constraints,
ArrayRef<StringRef> Clobbers,
ArrayRef<Expr*> Exprs,
SourceLocation EndLoc);
LabelDecl *GetOrCreateMSAsmLabel(StringRef ExternalLabelName,
SourceLocation Location,
bool AlwaysCreate);

VarDecl *BuildObjCExceptionDecl(TypeSourceInfo *TInfo, QualType ExceptionType,
SourceLocation StartLoc,
SourceLocation IdLoc, IdentifierInfo *Id,
bool Invalid = false);

Decl *ActOnObjCExceptionDecl(Scope *S, Declarator &D);

StmtResult ActOnObjCAtCatchStmt(SourceLocation AtLoc, SourceLocation RParen,
Decl *Parm, Stmt *Body);

StmtResult ActOnObjCAtFinallyStmt(SourceLocation AtLoc, Stmt *Body);

StmtResult ActOnObjCAtTryStmt(SourceLocation AtLoc, Stmt *Try,
MultiStmtArg Catch, Stmt *Finally);

StmtResult BuildObjCAtThrowStmt(SourceLocation AtLoc, Expr *Throw);
StmtResult ActOnObjCAtThrowStmt(SourceLocation AtLoc, Expr *Throw,
Scope *CurScope);
ExprResult ActOnObjCAtSynchronizedOperand(SourceLocation atLoc,
Expr *operand);
StmtResult ActOnObjCAtSynchronizedStmt(SourceLocation AtLoc,
Expr *SynchExpr,
Stmt *SynchBody);

StmtResult ActOnObjCAutoreleasePoolStmt(SourceLocation AtLoc, Stmt *Body);

VarDecl *BuildExceptionDeclaration(Scope *S, TypeSourceInfo *TInfo,
SourceLocation StartLoc,
SourceLocation IdLoc,
IdentifierInfo *Id);

Decl *ActOnExceptionDeclarator(Scope *S, Declarator &D);

StmtResult ActOnCXXCatchBlock(SourceLocation CatchLoc,
Decl *ExDecl, Stmt *HandlerBlock);
StmtResult ActOnCXXTryBlock(SourceLocation TryLoc, Stmt *TryBlock,
ArrayRef<Stmt *> Handlers);

StmtResult ActOnSEHTryBlock(bool IsCXXTry, 
SourceLocation TryLoc, Stmt *TryBlock,
Stmt *Handler);
StmtResult ActOnSEHExceptBlock(SourceLocation Loc,
Expr *FilterExpr,
Stmt *Block);
void ActOnStartSEHFinallyBlock();
void ActOnAbortSEHFinallyBlock();
StmtResult ActOnFinishSEHFinallyBlock(SourceLocation Loc, Stmt *Block);
StmtResult ActOnSEHLeaveStmt(SourceLocation Loc, Scope *CurScope);

void DiagnoseReturnInConstructorExceptionHandler(CXXTryStmt *TryBlock);

bool ShouldWarnIfUnusedFileScopedDecl(const DeclaratorDecl *D) const;

void MarkUnusedFileScopedDecl(const DeclaratorDecl *D);

void DiagnoseUnusedExprResult(const Stmt *S);
void DiagnoseUnusedNestedTypedefs(const RecordDecl *D);
void DiagnoseUnusedDecl(const NamedDecl *ND);

void DiagnoseEmptyStmtBody(SourceLocation StmtLoc,
const Stmt *Body,
unsigned DiagID);

void DiagnoseEmptyLoopBody(const Stmt *S,
const Stmt *PossibleBody);

void DiagnoseSelfMove(const Expr *LHSExpr, const Expr *RHSExpr,
SourceLocation OpLoc);

void diagnoseNullableToNonnullConversion(QualType DstType, QualType SrcType,
SourceLocation Loc);

void diagnoseZeroToNullptrConversion(CastKind Kind, const Expr *E);

ParsingDeclState PushParsingDeclaration(sema::DelayedDiagnosticPool &pool) {
return DelayedDiagnostics.push(pool);
}
void PopParsingDeclaration(ParsingDeclState state, Decl *decl);

typedef ProcessingContextState ParsingClassState;
ParsingClassState PushParsingClass() {
return DelayedDiagnostics.pushUndelayed();
}
void PopParsingClass(ParsingClassState state) {
DelayedDiagnostics.popUndelayed(state);
}

void redelayDiagnostics(sema::DelayedDiagnosticPool &pool);

void DiagnoseAvailabilityOfDecl(NamedDecl *D, ArrayRef<SourceLocation> Locs,
const ObjCInterfaceDecl *UnknownObjCClass,
bool ObjCPropertyAccess,
bool AvoidPartialAvailabilityChecks = false);

bool makeUnavailableInSystemHeader(SourceLocation loc,
UnavailableAttr::ImplicitReason reason);

void DiagnoseUnguardedAvailabilityViolations(Decl *FD);


bool CanUseDecl(NamedDecl *D, bool TreatUnavailableAsInvalid);
bool DiagnoseUseOfDecl(NamedDecl *D, ArrayRef<SourceLocation> Locs,
const ObjCInterfaceDecl *UnknownObjCClass = nullptr,
bool ObjCPropertyAccess = false,
bool AvoidPartialAvailabilityChecks = false);
void NoteDeletedFunction(FunctionDecl *FD);
void NoteDeletedInheritingConstructor(CXXConstructorDecl *CD);
std::string getDeletedOrUnavailableSuffix(const FunctionDecl *FD);
bool DiagnosePropertyAccessorMismatch(ObjCPropertyDecl *PD,
ObjCMethodDecl *Getter,
SourceLocation Loc);
void DiagnoseSentinelCalls(NamedDecl *D, SourceLocation Loc,
ArrayRef<Expr *> Args);

void PushExpressionEvaluationContext(
ExpressionEvaluationContext NewContext, Decl *LambdaContextDecl = nullptr,
ExpressionEvaluationContextRecord::ExpressionKind Type =
ExpressionEvaluationContextRecord::EK_Other);
enum ReuseLambdaContextDecl_t { ReuseLambdaContextDecl };
void PushExpressionEvaluationContext(
ExpressionEvaluationContext NewContext, ReuseLambdaContextDecl_t,
ExpressionEvaluationContextRecord::ExpressionKind Type =
ExpressionEvaluationContextRecord::EK_Other);
void PopExpressionEvaluationContext();

void DiscardCleanupsInEvaluationContext();

ExprResult TransformToPotentiallyEvaluated(Expr *E);
ExprResult HandleExprEvaluationContextForTypeof(Expr *E);

ExprResult ActOnConstantExpression(ExprResult Res);

void MarkAnyDeclReferenced(SourceLocation Loc, Decl *D, bool MightBeOdrUse);
void MarkFunctionReferenced(SourceLocation Loc, FunctionDecl *Func,
bool MightBeOdrUse = true);
void MarkVariableReferenced(SourceLocation Loc, VarDecl *Var);
void MarkDeclRefReferenced(DeclRefExpr *E, const Expr *Base = nullptr);
void MarkMemberReferenced(MemberExpr *E);

void UpdateMarkingForLValueToRValue(Expr *E);
void CleanupVarDeclMarking();

enum TryCaptureKind {
TryCapture_Implicit, TryCapture_ExplicitByVal, TryCapture_ExplicitByRef
};

bool tryCaptureVariable(VarDecl *Var, SourceLocation Loc, TryCaptureKind Kind,
SourceLocation EllipsisLoc, bool BuildAndDiagnose,
QualType &CaptureType,
QualType &DeclRefType,
const unsigned *const FunctionScopeIndexToStopAt);

bool tryCaptureVariable(VarDecl *Var, SourceLocation Loc,
TryCaptureKind Kind = TryCapture_Implicit,
SourceLocation EllipsisLoc = SourceLocation());

bool NeedToCaptureVariable(VarDecl *Var, SourceLocation Loc);

QualType getCapturedDeclRefType(VarDecl *Var, SourceLocation Loc);

void MarkDeclarationsReferencedInType(SourceLocation Loc, QualType T);
void MarkDeclarationsReferencedInExpr(Expr *E,
bool SkipLocalVariables = false);

bool tryToRecoverWithCall(ExprResult &E, const PartialDiagnostic &PD,
bool ForceComplain = false,
bool (*IsPlausibleResult)(QualType) = nullptr);

bool tryExprAsCall(Expr &E, QualType &ZeroArgCallReturnTy,
UnresolvedSetImpl &NonTemplateOverloads);

bool DiagRuntimeBehavior(SourceLocation Loc, const Stmt *Statement,
const PartialDiagnostic &PD);

SourceRange getExprRange(Expr *E) const;

ExprResult ActOnIdExpression(
Scope *S, CXXScopeSpec &SS, SourceLocation TemplateKWLoc,
UnqualifiedId &Id, bool HasTrailingLParen, bool IsAddressOfOperand,
std::unique_ptr<CorrectionCandidateCallback> CCC = nullptr,
bool IsInlineAsmIdentifier = false, Token *KeywordReplacement = nullptr);

void DecomposeUnqualifiedId(const UnqualifiedId &Id,
TemplateArgumentListInfo &Buffer,
DeclarationNameInfo &NameInfo,
const TemplateArgumentListInfo *&TemplateArgs);

bool
DiagnoseEmptyLookup(Scope *S, CXXScopeSpec &SS, LookupResult &R,
std::unique_ptr<CorrectionCandidateCallback> CCC,
TemplateArgumentListInfo *ExplicitTemplateArgs = nullptr,
ArrayRef<Expr *> Args = None, TypoExpr **Out = nullptr);

ExprResult LookupInObjCMethod(LookupResult &LookUp, Scope *S,
IdentifierInfo *II,
bool AllowBuiltinCreation=false);

ExprResult ActOnDependentIdExpression(const CXXScopeSpec &SS,
SourceLocation TemplateKWLoc,
const DeclarationNameInfo &NameInfo,
bool isAddressOfOperand,
const TemplateArgumentListInfo *TemplateArgs);

ExprResult BuildDeclRefExpr(ValueDecl *D, QualType Ty,
ExprValueKind VK,
SourceLocation Loc,
const CXXScopeSpec *SS = nullptr);
ExprResult
BuildDeclRefExpr(ValueDecl *D, QualType Ty, ExprValueKind VK,
const DeclarationNameInfo &NameInfo,
const CXXScopeSpec *SS = nullptr,
NamedDecl *FoundD = nullptr,
const TemplateArgumentListInfo *TemplateArgs = nullptr);
ExprResult
BuildAnonymousStructUnionMemberReference(
const CXXScopeSpec &SS,
SourceLocation nameLoc,
IndirectFieldDecl *indirectField,
DeclAccessPair FoundDecl = DeclAccessPair::make(nullptr, AS_none),
Expr *baseObjectExpr = nullptr,
SourceLocation opLoc = SourceLocation());

ExprResult BuildPossibleImplicitMemberExpr(const CXXScopeSpec &SS,
SourceLocation TemplateKWLoc,
LookupResult &R,
const TemplateArgumentListInfo *TemplateArgs,
const Scope *S);
ExprResult BuildImplicitMemberExpr(const CXXScopeSpec &SS,
SourceLocation TemplateKWLoc,
LookupResult &R,
const TemplateArgumentListInfo *TemplateArgs,
bool IsDefiniteInstance,
const Scope *S);
bool UseArgumentDependentLookup(const CXXScopeSpec &SS,
const LookupResult &R,
bool HasTrailingLParen);

ExprResult
BuildQualifiedDeclarationNameExpr(CXXScopeSpec &SS,
const DeclarationNameInfo &NameInfo,
bool IsAddressOfOperand, const Scope *S,
TypeSourceInfo **RecoveryTSI = nullptr);

ExprResult BuildDependentDeclRefExpr(const CXXScopeSpec &SS,
SourceLocation TemplateKWLoc,
const DeclarationNameInfo &NameInfo,
const TemplateArgumentListInfo *TemplateArgs);

ExprResult BuildDeclarationNameExpr(const CXXScopeSpec &SS,
LookupResult &R,
bool NeedsADL,
bool AcceptInvalidDecl = false);
ExprResult BuildDeclarationNameExpr(
const CXXScopeSpec &SS, const DeclarationNameInfo &NameInfo, NamedDecl *D,
NamedDecl *FoundD = nullptr,
const TemplateArgumentListInfo *TemplateArgs = nullptr,
bool AcceptInvalidDecl = false);

ExprResult BuildLiteralOperatorCall(LookupResult &R,
DeclarationNameInfo &SuffixInfo,
ArrayRef<Expr *> Args,
SourceLocation LitEndLoc,
TemplateArgumentListInfo *ExplicitTemplateArgs = nullptr);

ExprResult BuildPredefinedExpr(SourceLocation Loc,
PredefinedExpr::IdentType IT);
ExprResult ActOnPredefinedExpr(SourceLocation Loc, tok::TokenKind Kind);
ExprResult ActOnIntegerConstant(SourceLocation Loc, uint64_t Val);

bool CheckLoopHintExpr(Expr *E, SourceLocation Loc);

ExprResult ActOnNumericConstant(const Token &Tok, Scope *UDLScope = nullptr);
ExprResult ActOnCharacterConstant(const Token &Tok,
Scope *UDLScope = nullptr);
ExprResult ActOnParenExpr(SourceLocation L, SourceLocation R, Expr *E);
ExprResult ActOnParenListExpr(SourceLocation L,
SourceLocation R,
MultiExprArg Val);

ExprResult ActOnStringLiteral(ArrayRef<Token> StringToks,
Scope *UDLScope = nullptr);

ExprResult ActOnGenericSelectionExpr(SourceLocation KeyLoc,
SourceLocation DefaultLoc,
SourceLocation RParenLoc,
Expr *ControllingExpr,
ArrayRef<ParsedType> ArgTypes,
ArrayRef<Expr *> ArgExprs);
ExprResult CreateGenericSelectionExpr(SourceLocation KeyLoc,
SourceLocation DefaultLoc,
SourceLocation RParenLoc,
Expr *ControllingExpr,
ArrayRef<TypeSourceInfo *> Types,
ArrayRef<Expr *> Exprs);

ExprResult CreateBuiltinUnaryOp(SourceLocation OpLoc, UnaryOperatorKind Opc,
Expr *InputExpr);
ExprResult BuildUnaryOp(Scope *S, SourceLocation OpLoc,
UnaryOperatorKind Opc, Expr *Input);
ExprResult ActOnUnaryOp(Scope *S, SourceLocation OpLoc,
tok::TokenKind Op, Expr *Input);

bool isQualifiedMemberAccess(Expr *E);
QualType CheckAddressOfOperand(ExprResult &Operand, SourceLocation OpLoc);

ExprResult CreateUnaryExprOrTypeTraitExpr(TypeSourceInfo *TInfo,
SourceLocation OpLoc,
UnaryExprOrTypeTrait ExprKind,
SourceRange R);
ExprResult CreateUnaryExprOrTypeTraitExpr(Expr *E, SourceLocation OpLoc,
UnaryExprOrTypeTrait ExprKind);
ExprResult
ActOnUnaryExprOrTypeTraitExpr(SourceLocation OpLoc,
UnaryExprOrTypeTrait ExprKind,
bool IsType, void *TyOrEx,
SourceRange ArgRange);

ExprResult CheckPlaceholderExpr(Expr *E);
bool CheckVecStepExpr(Expr *E);

bool CheckUnaryExprOrTypeTraitOperand(Expr *E, UnaryExprOrTypeTrait ExprKind);
bool CheckUnaryExprOrTypeTraitOperand(QualType ExprType, SourceLocation OpLoc,
SourceRange ExprRange,
UnaryExprOrTypeTrait ExprKind);
ExprResult ActOnSizeofParameterPackExpr(Scope *S,
SourceLocation OpLoc,
IdentifierInfo &Name,
SourceLocation NameLoc,
SourceLocation RParenLoc);
ExprResult ActOnPostfixUnaryOp(Scope *S, SourceLocation OpLoc,
tok::TokenKind Kind, Expr *Input);

ExprResult ActOnArraySubscriptExpr(Scope *S, Expr *Base, SourceLocation LLoc,
Expr *Idx, SourceLocation RLoc);
ExprResult CreateBuiltinArraySubscriptExpr(Expr *Base, SourceLocation LLoc,
Expr *Idx, SourceLocation RLoc);
ExprResult ActOnOMPArraySectionExpr(Expr *Base, SourceLocation LBLoc,
Expr *LowerBound, SourceLocation ColonLoc,
Expr *Length, SourceLocation RBLoc);

struct ActOnMemberAccessExtraArgs {
Scope *S;
UnqualifiedId &Id;
Decl *ObjCImpDecl;
};

ExprResult BuildMemberReferenceExpr(
Expr *Base, QualType BaseType, SourceLocation OpLoc, bool IsArrow,
CXXScopeSpec &SS, SourceLocation TemplateKWLoc,
NamedDecl *FirstQualifierInScope, const DeclarationNameInfo &NameInfo,
const TemplateArgumentListInfo *TemplateArgs,
const Scope *S,
ActOnMemberAccessExtraArgs *ExtraArgs = nullptr);

ExprResult
BuildMemberReferenceExpr(Expr *Base, QualType BaseType, SourceLocation OpLoc,
bool IsArrow, const CXXScopeSpec &SS,
SourceLocation TemplateKWLoc,
NamedDecl *FirstQualifierInScope, LookupResult &R,
const TemplateArgumentListInfo *TemplateArgs,
const Scope *S,
bool SuppressQualifierCheck = false,
ActOnMemberAccessExtraArgs *ExtraArgs = nullptr);

ExprResult BuildFieldReferenceExpr(Expr *BaseExpr, bool IsArrow,
SourceLocation OpLoc,
const CXXScopeSpec &SS, FieldDecl *Field,
DeclAccessPair FoundDecl,
const DeclarationNameInfo &MemberNameInfo);

ExprResult PerformMemberExprBaseConversion(Expr *Base, bool IsArrow);

bool CheckQualifiedMemberReference(Expr *BaseExpr, QualType BaseType,
const CXXScopeSpec &SS,
const LookupResult &R);

ExprResult ActOnDependentMemberExpr(Expr *Base, QualType BaseType,
bool IsArrow, SourceLocation OpLoc,
const CXXScopeSpec &SS,
SourceLocation TemplateKWLoc,
NamedDecl *FirstQualifierInScope,
const DeclarationNameInfo &NameInfo,
const TemplateArgumentListInfo *TemplateArgs);

ExprResult ActOnMemberAccessExpr(Scope *S, Expr *Base,
SourceLocation OpLoc,
tok::TokenKind OpKind,
CXXScopeSpec &SS,
SourceLocation TemplateKWLoc,
UnqualifiedId &Member,
Decl *ObjCImpDecl);

void ActOnDefaultCtorInitializers(Decl *CDtorDecl);
bool ConvertArgumentsForCall(CallExpr *Call, Expr *Fn,
FunctionDecl *FDecl,
const FunctionProtoType *Proto,
ArrayRef<Expr *> Args,
SourceLocation RParenLoc,
bool ExecConfig = false);
void CheckStaticArrayArgument(SourceLocation CallLoc,
ParmVarDecl *Param,
const Expr *ArgExpr);

ExprResult ActOnCallExpr(Scope *S, Expr *Fn, SourceLocation LParenLoc,
MultiExprArg ArgExprs, SourceLocation RParenLoc,
Expr *ExecConfig = nullptr,
bool IsExecConfig = false);
ExprResult BuildResolvedCallExpr(Expr *Fn, NamedDecl *NDecl,
SourceLocation LParenLoc,
ArrayRef<Expr *> Arg,
SourceLocation RParenLoc,
Expr *Config = nullptr,
bool IsExecConfig = false);

ExprResult ActOnCUDAExecConfigExpr(Scope *S, SourceLocation LLLLoc,
MultiExprArg ExecConfig,
SourceLocation GGGLoc);

ExprResult ActOnCastExpr(Scope *S, SourceLocation LParenLoc,
Declarator &D, ParsedType &Ty,
SourceLocation RParenLoc, Expr *CastExpr);
ExprResult BuildCStyleCastExpr(SourceLocation LParenLoc,
TypeSourceInfo *Ty,
SourceLocation RParenLoc,
Expr *Op);
CastKind PrepareScalarCast(ExprResult &src, QualType destType);

ExprResult BuildVectorLiteral(SourceLocation LParenLoc,
SourceLocation RParenLoc, Expr *E,
TypeSourceInfo *TInfo);

ExprResult MaybeConvertParenListExprToParenExpr(Scope *S, Expr *ME);

ExprResult ActOnCompoundLiteral(SourceLocation LParenLoc,
ParsedType Ty,
SourceLocation RParenLoc,
Expr *InitExpr);

ExprResult BuildCompoundLiteralExpr(SourceLocation LParenLoc,
TypeSourceInfo *TInfo,
SourceLocation RParenLoc,
Expr *LiteralExpr);

ExprResult ActOnInitList(SourceLocation LBraceLoc,
MultiExprArg InitArgList,
SourceLocation RBraceLoc);

ExprResult ActOnDesignatedInitializer(Designation &Desig,
SourceLocation Loc,
bool GNUSyntax,
ExprResult Init);

private:
static BinaryOperatorKind ConvertTokenKindToBinaryOpcode(tok::TokenKind Kind);

public:
ExprResult ActOnBinOp(Scope *S, SourceLocation TokLoc,
tok::TokenKind Kind, Expr *LHSExpr, Expr *RHSExpr);
ExprResult BuildBinOp(Scope *S, SourceLocation OpLoc,
BinaryOperatorKind Opc, Expr *LHSExpr, Expr *RHSExpr);
ExprResult CreateBuiltinBinOp(SourceLocation OpLoc, BinaryOperatorKind Opc,
Expr *LHSExpr, Expr *RHSExpr);

void DiagnoseCommaOperator(const Expr *LHS, SourceLocation Loc);

ExprResult ActOnConditionalOp(SourceLocation QuestionLoc,
SourceLocation ColonLoc,
Expr *CondExpr, Expr *LHSExpr, Expr *RHSExpr);

ExprResult ActOnAddrLabel(SourceLocation OpLoc, SourceLocation LabLoc,
LabelDecl *TheDecl);

void ActOnStartStmtExpr();
ExprResult ActOnStmtExpr(SourceLocation LPLoc, Stmt *SubStmt,
SourceLocation RPLoc); 
void ActOnStmtExprError();

struct OffsetOfComponent {
SourceLocation LocStart, LocEnd;
bool isBrackets;  
union {
IdentifierInfo *IdentInfo;
Expr *E;
} U;
};

ExprResult BuildBuiltinOffsetOf(SourceLocation BuiltinLoc,
TypeSourceInfo *TInfo,
ArrayRef<OffsetOfComponent> Components,
SourceLocation RParenLoc);
ExprResult ActOnBuiltinOffsetOf(Scope *S,
SourceLocation BuiltinLoc,
SourceLocation TypeLoc,
ParsedType ParsedArgTy,
ArrayRef<OffsetOfComponent> Components,
SourceLocation RParenLoc);

ExprResult ActOnChooseExpr(SourceLocation BuiltinLoc,
Expr *CondExpr, Expr *LHSExpr,
Expr *RHSExpr, SourceLocation RPLoc);

ExprResult ActOnVAArg(SourceLocation BuiltinLoc, Expr *E, ParsedType Ty,
SourceLocation RPLoc);
ExprResult BuildVAArgExpr(SourceLocation BuiltinLoc, Expr *E,
TypeSourceInfo *TInfo, SourceLocation RPLoc);

ExprResult ActOnGNUNullExpr(SourceLocation TokenLoc);

bool CheckCaseExpression(Expr *E);

enum IfExistsResult {
IER_Exists,

IER_DoesNotExist,

IER_Dependent,

IER_Error
};

IfExistsResult
CheckMicrosoftIfExistsSymbol(Scope *S, CXXScopeSpec &SS,
const DeclarationNameInfo &TargetNameInfo);

IfExistsResult
CheckMicrosoftIfExistsSymbol(Scope *S, SourceLocation KeywordLoc,
bool IsIfExists, CXXScopeSpec &SS,
UnqualifiedId &Name);

StmtResult BuildMSDependentExistsStmt(SourceLocation KeywordLoc,
bool IsIfExists,
NestedNameSpecifierLoc QualifierLoc,
DeclarationNameInfo NameInfo,
Stmt *Nested);
StmtResult ActOnMSDependentExistsStmt(SourceLocation KeywordLoc,
bool IsIfExists,
CXXScopeSpec &SS, UnqualifiedId &Name,
Stmt *Nested);


void ActOnBlockStart(SourceLocation CaretLoc, Scope *CurScope);

void ActOnBlockArguments(SourceLocation CaretLoc, Declarator &ParamInfo,
Scope *CurScope);

void ActOnBlockError(SourceLocation CaretLoc, Scope *CurScope);

ExprResult ActOnBlockStmtExpr(SourceLocation CaretLoc, Stmt *Body,
Scope *CurScope);


ExprResult ActOnConvertVectorExpr(Expr *E, ParsedType ParsedDestTy,
SourceLocation BuiltinLoc,
SourceLocation RParenLoc);


ExprResult ActOnAsTypeExpr(Expr *E, ParsedType ParsedDestTy,
SourceLocation BuiltinLoc,
SourceLocation RParenLoc);


Decl *ActOnStartNamespaceDef(Scope *S, SourceLocation InlineLoc,
SourceLocation NamespaceLoc,
SourceLocation IdentLoc, IdentifierInfo *Ident,
SourceLocation LBrace,
const ParsedAttributesView &AttrList,
UsingDirectiveDecl *&UsingDecl);
void ActOnFinishNamespaceDef(Decl *Dcl, SourceLocation RBrace);

NamespaceDecl *getStdNamespace() const;
NamespaceDecl *getOrCreateStdNamespace();

NamespaceDecl *lookupStdExperimentalNamespace();

CXXRecordDecl *getStdBadAlloc() const;
EnumDecl *getStdAlignValT() const;

private:
llvm::SmallBitVector FullyCheckedComparisonCategories;

public:
QualType CheckComparisonCategoryType(ComparisonCategoryType Kind,
SourceLocation Loc);

bool isStdInitializerList(QualType Ty, QualType *Element);

QualType BuildStdInitializerList(QualType Element, SourceLocation Loc);

bool isInitListConstructor(const FunctionDecl *Ctor);

Decl *ActOnUsingDirective(Scope *CurScope, SourceLocation UsingLoc,
SourceLocation NamespcLoc, CXXScopeSpec &SS,
SourceLocation IdentLoc,
IdentifierInfo *NamespcName,
const ParsedAttributesView &AttrList);

void PushUsingDirective(Scope *S, UsingDirectiveDecl *UDir);

Decl *ActOnNamespaceAliasDef(Scope *CurScope,
SourceLocation NamespaceLoc,
SourceLocation AliasLoc,
IdentifierInfo *Alias,
CXXScopeSpec &SS,
SourceLocation IdentLoc,
IdentifierInfo *Ident);

void HideUsingShadowDecl(Scope *S, UsingShadowDecl *Shadow);
bool CheckUsingShadowDecl(UsingDecl *UD, NamedDecl *Target,
const LookupResult &PreviousDecls,
UsingShadowDecl *&PrevShadow);
UsingShadowDecl *BuildUsingShadowDecl(Scope *S, UsingDecl *UD,
NamedDecl *Target,
UsingShadowDecl *PrevDecl);

bool CheckUsingDeclRedeclaration(SourceLocation UsingLoc,
bool HasTypenameKeyword,
const CXXScopeSpec &SS,
SourceLocation NameLoc,
const LookupResult &Previous);
bool CheckUsingDeclQualifier(SourceLocation UsingLoc,
bool HasTypename,
const CXXScopeSpec &SS,
const DeclarationNameInfo &NameInfo,
SourceLocation NameLoc);

NamedDecl *BuildUsingDeclaration(
Scope *S, AccessSpecifier AS, SourceLocation UsingLoc,
bool HasTypenameKeyword, SourceLocation TypenameLoc, CXXScopeSpec &SS,
DeclarationNameInfo NameInfo, SourceLocation EllipsisLoc,
const ParsedAttributesView &AttrList, bool IsInstantiation);
NamedDecl *BuildUsingPackDecl(NamedDecl *InstantiatedFrom,
ArrayRef<NamedDecl *> Expansions);

bool CheckInheritingConstructorUsingDecl(UsingDecl *UD);

CXXConstructorDecl *
findInheritingConstructor(SourceLocation Loc, CXXConstructorDecl *BaseCtor,
ConstructorUsingShadowDecl *DerivedShadow);

Decl *ActOnUsingDeclaration(Scope *CurScope, AccessSpecifier AS,
SourceLocation UsingLoc,
SourceLocation TypenameLoc, CXXScopeSpec &SS,
UnqualifiedId &Name, SourceLocation EllipsisLoc,
const ParsedAttributesView &AttrList);
Decl *ActOnAliasDeclaration(Scope *CurScope, AccessSpecifier AS,
MultiTemplateParamsArg TemplateParams,
SourceLocation UsingLoc, UnqualifiedId &Name,
const ParsedAttributesView &AttrList,
TypeResult Type, Decl *DeclFromDeclSpec);

ExprResult
BuildCXXConstructExpr(SourceLocation ConstructLoc, QualType DeclInitType,
NamedDecl *FoundDecl,
CXXConstructorDecl *Constructor, MultiExprArg Exprs,
bool HadMultipleCandidates, bool IsListInitialization,
bool IsStdInitListInitialization,
bool RequiresZeroInit, unsigned ConstructKind,
SourceRange ParenRange);

ExprResult
BuildCXXConstructExpr(SourceLocation ConstructLoc, QualType DeclInitType,
CXXConstructorDecl *Constructor, bool Elidable,
MultiExprArg Exprs,
bool HadMultipleCandidates, bool IsListInitialization,
bool IsStdInitListInitialization,
bool RequiresZeroInit, unsigned ConstructKind,
SourceRange ParenRange);

ExprResult
BuildCXXConstructExpr(SourceLocation ConstructLoc, QualType DeclInitType,
NamedDecl *FoundDecl,
CXXConstructorDecl *Constructor, bool Elidable,
MultiExprArg Exprs, bool HadMultipleCandidates,
bool IsListInitialization,
bool IsStdInitListInitialization, bool RequiresZeroInit,
unsigned ConstructKind, SourceRange ParenRange);

ExprResult BuildCXXDefaultInitExpr(SourceLocation Loc, FieldDecl *Field);


bool CheckCXXDefaultArgExpr(SourceLocation CallLoc, FunctionDecl *FD,
ParmVarDecl *Param);

ExprResult BuildCXXDefaultArgExpr(SourceLocation CallLoc,
FunctionDecl *FD,
ParmVarDecl *Param);

void FinalizeVarWithDestructor(VarDecl *VD, const RecordType *DeclInitType);

class ImplicitExceptionSpecification {
Sema *Self;
ExceptionSpecificationType ComputedEST;
llvm::SmallPtrSet<CanQualType, 4> ExceptionsSeen;
SmallVector<QualType, 4> Exceptions;

void ClearExceptions() {
ExceptionsSeen.clear();
Exceptions.clear();
}

public:
explicit ImplicitExceptionSpecification(Sema &Self)
: Self(&Self), ComputedEST(EST_BasicNoexcept) {
if (!Self.getLangOpts().CPlusPlus11)
ComputedEST = EST_DynamicNone;
}

ExceptionSpecificationType getExceptionSpecType() const {
assert(!isComputedNoexcept(ComputedEST) &&
"noexcept(expr) should not be a possible result");
return ComputedEST;
}

unsigned size() const { return Exceptions.size(); }

const QualType *data() const { return Exceptions.data(); }

void CalledDecl(SourceLocation CallLoc, const CXXMethodDecl *Method);

void CalledExpr(Expr *E);

FunctionProtoType::ExceptionSpecInfo getExceptionSpec() const {
FunctionProtoType::ExceptionSpecInfo ESI;
ESI.Type = getExceptionSpecType();
if (ESI.Type == EST_Dynamic) {
ESI.Exceptions = Exceptions;
} else if (ESI.Type == EST_None) {
ESI.Type = EST_NoexceptFalse;
ESI.NoexceptExpr = Self->ActOnCXXBoolLiteral(SourceLocation(),
tok::kw_false).get();
}
return ESI;
}
};

ImplicitExceptionSpecification
ComputeDefaultedDefaultCtorExceptionSpec(SourceLocation Loc,
CXXMethodDecl *MD);

ImplicitExceptionSpecification
ComputeDefaultedCopyCtorExceptionSpec(CXXMethodDecl *MD);

ImplicitExceptionSpecification
ComputeDefaultedCopyAssignmentExceptionSpec(CXXMethodDecl *MD);

ImplicitExceptionSpecification
ComputeDefaultedMoveCtorExceptionSpec(CXXMethodDecl *MD);

ImplicitExceptionSpecification
ComputeDefaultedMoveAssignmentExceptionSpec(CXXMethodDecl *MD);

ImplicitExceptionSpecification
ComputeDefaultedDtorExceptionSpec(CXXMethodDecl *MD);

ImplicitExceptionSpecification
ComputeInheritingCtorExceptionSpec(SourceLocation Loc,
CXXConstructorDecl *CD);

void EvaluateImplicitExceptionSpec(SourceLocation Loc, CXXMethodDecl *MD);

ExprResult ActOnNoexceptSpec(SourceLocation NoexceptLoc, Expr *NoexceptExpr,
ExceptionSpecificationType &EST);

void checkExceptionSpecification(bool IsTopLevel,
ExceptionSpecificationType EST,
ArrayRef<ParsedType> DynamicExceptions,
ArrayRef<SourceRange> DynamicExceptionRanges,
Expr *NoexceptExpr,
SmallVectorImpl<QualType> &Exceptions,
FunctionProtoType::ExceptionSpecInfo &ESI);

bool isLibstdcxxEagerExceptionSpecHack(const Declarator &D);

void actOnDelayedExceptionSpecification(Decl *Method,
ExceptionSpecificationType EST,
SourceRange SpecificationRange,
ArrayRef<ParsedType> DynamicExceptions,
ArrayRef<SourceRange> DynamicExceptionRanges,
Expr *NoexceptExpr);

class InheritedConstructorInfo;

bool ShouldDeleteSpecialMember(CXXMethodDecl *MD, CXXSpecialMember CSM,
InheritedConstructorInfo *ICI = nullptr,
bool Diagnose = false);

CXXConstructorDecl *DeclareImplicitDefaultConstructor(
CXXRecordDecl *ClassDecl);

void DefineImplicitDefaultConstructor(SourceLocation CurrentLocation,
CXXConstructorDecl *Constructor);

CXXDestructorDecl *DeclareImplicitDestructor(CXXRecordDecl *ClassDecl);

void DefineImplicitDestructor(SourceLocation CurrentLocation,
CXXDestructorDecl *Destructor);

void AdjustDestructorExceptionSpec(CXXRecordDecl *ClassDecl,
CXXDestructorDecl *Destructor);

void DefineInheritingConstructor(SourceLocation UseLoc,
CXXConstructorDecl *Constructor);

CXXConstructorDecl *DeclareImplicitCopyConstructor(CXXRecordDecl *ClassDecl);

void DefineImplicitCopyConstructor(SourceLocation CurrentLocation,
CXXConstructorDecl *Constructor);

CXXConstructorDecl *DeclareImplicitMoveConstructor(CXXRecordDecl *ClassDecl);

void DefineImplicitMoveConstructor(SourceLocation CurrentLocation,
CXXConstructorDecl *Constructor);

CXXMethodDecl *DeclareImplicitCopyAssignment(CXXRecordDecl *ClassDecl);

void DefineImplicitCopyAssignment(SourceLocation CurrentLocation,
CXXMethodDecl *MethodDecl);

CXXMethodDecl *DeclareImplicitMoveAssignment(CXXRecordDecl *ClassDecl);

void DefineImplicitMoveAssignment(SourceLocation CurrentLocation,
CXXMethodDecl *MethodDecl);

void ForceDeclarationOfImplicitMembers(CXXRecordDecl *Class);

void CheckImplicitSpecialMemberDeclaration(Scope *S, FunctionDecl *FD);

bool isImplicitlyDeleted(FunctionDecl *FD);

bool checkThisInStaticMemberFunctionType(CXXMethodDecl *Method);

bool checkThisInStaticMemberFunctionExceptionSpec(CXXMethodDecl *Method);

bool checkThisInStaticMemberFunctionAttributes(CXXMethodDecl *Method);

ExprResult MaybeBindToTemporary(Expr *E);

bool CompleteConstructorCall(CXXConstructorDecl *Constructor,
MultiExprArg ArgsPtr,
SourceLocation Loc,
SmallVectorImpl<Expr*> &ConvertedArgs,
bool AllowExplicit = false,
bool IsListInitialization = false);

ParsedType getInheritingConstructorName(CXXScopeSpec &SS,
SourceLocation NameLoc,
IdentifierInfo &Name);

ParsedType getConstructorName(IdentifierInfo &II, SourceLocation NameLoc,
Scope *S, CXXScopeSpec &SS,
bool EnteringContext);
ParsedType getDestructorName(SourceLocation TildeLoc,
IdentifierInfo &II, SourceLocation NameLoc,
Scope *S, CXXScopeSpec &SS,
ParsedType ObjectType,
bool EnteringContext);

ParsedType getDestructorTypeForDecltype(const DeclSpec &DS,
ParsedType ObjectType);

void CheckCompatibleReinterpretCast(QualType SrcType, QualType DestType,
bool IsDereference, SourceRange Range);

ExprResult ActOnCXXNamedCast(SourceLocation OpLoc,
tok::TokenKind Kind,
SourceLocation LAngleBracketLoc,
Declarator &D,
SourceLocation RAngleBracketLoc,
SourceLocation LParenLoc,
Expr *E,
SourceLocation RParenLoc);

ExprResult BuildCXXNamedCast(SourceLocation OpLoc,
tok::TokenKind Kind,
TypeSourceInfo *Ty,
Expr *E,
SourceRange AngleBrackets,
SourceRange Parens);

ExprResult BuildCXXTypeId(QualType TypeInfoType,
SourceLocation TypeidLoc,
TypeSourceInfo *Operand,
SourceLocation RParenLoc);
ExprResult BuildCXXTypeId(QualType TypeInfoType,
SourceLocation TypeidLoc,
Expr *Operand,
SourceLocation RParenLoc);

ExprResult ActOnCXXTypeid(SourceLocation OpLoc,
SourceLocation LParenLoc, bool isType,
void *TyOrExpr,
SourceLocation RParenLoc);

ExprResult BuildCXXUuidof(QualType TypeInfoType,
SourceLocation TypeidLoc,
TypeSourceInfo *Operand,
SourceLocation RParenLoc);
ExprResult BuildCXXUuidof(QualType TypeInfoType,
SourceLocation TypeidLoc,
Expr *Operand,
SourceLocation RParenLoc);

ExprResult ActOnCXXUuidof(SourceLocation OpLoc,
SourceLocation LParenLoc, bool isType,
void *TyOrExpr,
SourceLocation RParenLoc);

ExprResult ActOnCXXFoldExpr(SourceLocation LParenLoc, Expr *LHS,
tok::TokenKind Operator,
SourceLocation EllipsisLoc, Expr *RHS,
SourceLocation RParenLoc);
ExprResult BuildCXXFoldExpr(SourceLocation LParenLoc, Expr *LHS,
BinaryOperatorKind Operator,
SourceLocation EllipsisLoc, Expr *RHS,
SourceLocation RParenLoc);
ExprResult BuildEmptyCXXFoldExpr(SourceLocation EllipsisLoc,
BinaryOperatorKind Operator);

ExprResult ActOnCXXThis(SourceLocation loc);

QualType getCurrentThisType();

QualType CXXThisTypeOverride;

class CXXThisScopeRAII {
Sema &S;
QualType OldCXXThisTypeOverride;
bool Enabled;

public:
CXXThisScopeRAII(Sema &S, Decl *ContextDecl, unsigned CXXThisTypeQuals,
bool Enabled = true);

~CXXThisScopeRAII();
};

bool CheckCXXThisCapture(SourceLocation Loc, bool Explicit = false,
bool BuildAndDiagnose = true,
const unsigned *const FunctionScopeIndexToStopAt = nullptr,
bool ByCopy = false);

bool isThisOutsideMemberFunctionBody(QualType BaseType);

ExprResult ActOnCXXBoolLiteral(SourceLocation OpLoc, tok::TokenKind Kind);


ExprResult ActOnObjCBoolLiteral(SourceLocation OpLoc, tok::TokenKind Kind);

ExprResult
ActOnObjCAvailabilityCheckExpr(llvm::ArrayRef<AvailabilitySpec> AvailSpecs,
SourceLocation AtLoc, SourceLocation RParen);

ExprResult ActOnCXXNullPtrLiteral(SourceLocation Loc);

ExprResult ActOnCXXThrow(Scope *S, SourceLocation OpLoc, Expr *expr);
ExprResult BuildCXXThrow(SourceLocation OpLoc, Expr *Ex,
bool IsThrownVarInScope);
bool CheckCXXThrowOperand(SourceLocation ThrowLoc, QualType ThrowTy, Expr *E);

ExprResult ActOnCXXTypeConstructExpr(ParsedType TypeRep,
SourceLocation LParenOrBraceLoc,
MultiExprArg Exprs,
SourceLocation RParenOrBraceLoc,
bool ListInitialization);

ExprResult BuildCXXTypeConstructExpr(TypeSourceInfo *Type,
SourceLocation LParenLoc,
MultiExprArg Exprs,
SourceLocation RParenLoc,
bool ListInitialization);

ExprResult ActOnCXXNew(SourceLocation StartLoc, bool UseGlobal,
SourceLocation PlacementLParen,
MultiExprArg PlacementArgs,
SourceLocation PlacementRParen,
SourceRange TypeIdParens, Declarator &D,
Expr *Initializer);
ExprResult BuildCXXNew(SourceRange Range, bool UseGlobal,
SourceLocation PlacementLParen,
MultiExprArg PlacementArgs,
SourceLocation PlacementRParen,
SourceRange TypeIdParens,
QualType AllocType,
TypeSourceInfo *AllocTypeInfo,
Expr *ArraySize,
SourceRange DirectInitRange,
Expr *Initializer);

bool CheckAllocatedType(QualType AllocType, SourceLocation Loc,
SourceRange R);

enum AllocationFunctionScope {
AFS_Global,
AFS_Class,
AFS_Both
};

bool FindAllocationFunctions(SourceLocation StartLoc, SourceRange Range,
AllocationFunctionScope NewScope,
AllocationFunctionScope DeleteScope,
QualType AllocType, bool IsArray,
bool &PassAlignment, MultiExprArg PlaceArgs,
FunctionDecl *&OperatorNew,
FunctionDecl *&OperatorDelete,
bool Diagnose = true);
void DeclareGlobalNewDelete();
void DeclareGlobalAllocationFunction(DeclarationName Name, QualType Return,
ArrayRef<QualType> Params);

bool FindDeallocationFunction(SourceLocation StartLoc, CXXRecordDecl *RD,
DeclarationName Name, FunctionDecl* &Operator,
bool Diagnose = true);
FunctionDecl *FindUsualDeallocationFunction(SourceLocation StartLoc,
bool CanProvideSize,
bool Overaligned,
DeclarationName Name);
FunctionDecl *FindDeallocationFunctionForDestructor(SourceLocation StartLoc,
CXXRecordDecl *RD);

ExprResult ActOnCXXDelete(SourceLocation StartLoc,
bool UseGlobal, bool ArrayForm,
Expr *Operand);
void CheckVirtualDtorCall(CXXDestructorDecl *dtor, SourceLocation Loc,
bool IsDelete, bool CallCanBeVirtual,
bool WarnOnNonAbstractTypes,
SourceLocation DtorLoc);

ExprResult ActOnNoexceptExpr(SourceLocation KeyLoc, SourceLocation LParen,
Expr *Operand, SourceLocation RParen);
ExprResult BuildCXXNoexceptExpr(SourceLocation KeyLoc, Expr *Operand,
SourceLocation RParen);

ExprResult ActOnTypeTrait(TypeTrait Kind, SourceLocation KWLoc,
ArrayRef<ParsedType> Args,
SourceLocation RParenLoc);
ExprResult BuildTypeTrait(TypeTrait Kind, SourceLocation KWLoc,
ArrayRef<TypeSourceInfo *> Args,
SourceLocation RParenLoc);

ExprResult ActOnArrayTypeTrait(ArrayTypeTrait ATT,
SourceLocation KWLoc,
ParsedType LhsTy,
Expr *DimExpr,
SourceLocation RParen);

ExprResult BuildArrayTypeTrait(ArrayTypeTrait ATT,
SourceLocation KWLoc,
TypeSourceInfo *TSInfo,
Expr *DimExpr,
SourceLocation RParen);

ExprResult ActOnExpressionTrait(ExpressionTrait OET,
SourceLocation KWLoc,
Expr *Queried,
SourceLocation RParen);

ExprResult BuildExpressionTrait(ExpressionTrait OET,
SourceLocation KWLoc,
Expr *Queried,
SourceLocation RParen);

ExprResult ActOnStartCXXMemberReference(Scope *S,
Expr *Base,
SourceLocation OpLoc,
tok::TokenKind OpKind,
ParsedType &ObjectType,
bool &MayBePseudoDestructor);

ExprResult BuildPseudoDestructorExpr(Expr *Base,
SourceLocation OpLoc,
tok::TokenKind OpKind,
const CXXScopeSpec &SS,
TypeSourceInfo *ScopeType,
SourceLocation CCLoc,
SourceLocation TildeLoc,
PseudoDestructorTypeStorage DestroyedType);

ExprResult ActOnPseudoDestructorExpr(Scope *S, Expr *Base,
SourceLocation OpLoc,
tok::TokenKind OpKind,
CXXScopeSpec &SS,
UnqualifiedId &FirstTypeName,
SourceLocation CCLoc,
SourceLocation TildeLoc,
UnqualifiedId &SecondTypeName);

ExprResult ActOnPseudoDestructorExpr(Scope *S, Expr *Base,
SourceLocation OpLoc,
tok::TokenKind OpKind,
SourceLocation TildeLoc,
const DeclSpec& DS);

Expr *MaybeCreateExprWithCleanups(Expr *SubExpr);
Stmt *MaybeCreateStmtWithCleanups(Stmt *SubStmt);
ExprResult MaybeCreateExprWithCleanups(ExprResult SubExpr);

MaterializeTemporaryExpr *
CreateMaterializeTemporaryExpr(QualType T, Expr *Temporary,
bool BoundToLvalueReference);

ExprResult ActOnFinishFullExpr(Expr *Expr) {
return ActOnFinishFullExpr(Expr, Expr ? Expr->getExprLoc()
: SourceLocation());
}
ExprResult ActOnFinishFullExpr(Expr *Expr, SourceLocation CC,
bool DiscardedValue = false,
bool IsConstexpr = false,
bool IsLambdaInitCaptureInitializer = false);
StmtResult ActOnFinishFullStmt(Stmt *Stmt);

bool RequireCompleteDeclContext(CXXScopeSpec &SS, DeclContext *DC);

DeclContext *computeDeclContext(QualType T);
DeclContext *computeDeclContext(const CXXScopeSpec &SS,
bool EnteringContext = false);
bool isDependentScopeSpecifier(const CXXScopeSpec &SS);
CXXRecordDecl *getCurrentInstantiationOf(NestedNameSpecifier *NNS);

bool ActOnCXXGlobalScopeSpecifier(SourceLocation CCLoc, CXXScopeSpec &SS);

bool ActOnSuperScopeSpecifier(SourceLocation SuperLoc,
SourceLocation ColonColonLoc, CXXScopeSpec &SS);

bool isAcceptableNestedNameSpecifier(const NamedDecl *SD,
bool *CanCorrect = nullptr);
NamedDecl *FindFirstQualifierInScope(Scope *S, NestedNameSpecifier *NNS);

struct NestedNameSpecInfo {
ParsedType ObjectType;

IdentifierInfo *Identifier;

SourceLocation IdentifierLoc;

SourceLocation CCLoc;

NestedNameSpecInfo(IdentifierInfo *II, SourceLocation IdLoc,
SourceLocation ColonColonLoc, ParsedType ObjectType = ParsedType())
: ObjectType(ObjectType), Identifier(II), IdentifierLoc(IdLoc),
CCLoc(ColonColonLoc) {
}

NestedNameSpecInfo(IdentifierInfo *II, SourceLocation IdLoc,
SourceLocation ColonColonLoc, QualType ObjectType)
: ObjectType(ParsedType::make(ObjectType)), Identifier(II),
IdentifierLoc(IdLoc), CCLoc(ColonColonLoc) {
}
};

bool isNonTypeNestedNameSpecifier(Scope *S, CXXScopeSpec &SS,
NestedNameSpecInfo &IdInfo);

bool BuildCXXNestedNameSpecifier(Scope *S,
NestedNameSpecInfo &IdInfo,
bool EnteringContext,
CXXScopeSpec &SS,
NamedDecl *ScopeLookupResult,
bool ErrorRecoveryLookup,
bool *IsCorrectedToColon = nullptr,
bool OnlyNamespace = false);

bool ActOnCXXNestedNameSpecifier(Scope *S,
NestedNameSpecInfo &IdInfo,
bool EnteringContext,
CXXScopeSpec &SS,
bool ErrorRecoveryLookup = false,
bool *IsCorrectedToColon = nullptr,
bool OnlyNamespace = false);

ExprResult ActOnDecltypeExpression(Expr *E);

bool ActOnCXXNestedNameSpecifierDecltype(CXXScopeSpec &SS,
const DeclSpec &DS,
SourceLocation ColonColonLoc);

bool IsInvalidUnlessNestedName(Scope *S, CXXScopeSpec &SS,
NestedNameSpecInfo &IdInfo,
bool EnteringContext);

bool ActOnCXXNestedNameSpecifier(Scope *S,
CXXScopeSpec &SS,
SourceLocation TemplateKWLoc,
TemplateTy TemplateName,
SourceLocation TemplateNameLoc,
SourceLocation LAngleLoc,
ASTTemplateArgsPtr TemplateArgs,
SourceLocation RAngleLoc,
SourceLocation CCLoc,
bool EnteringContext);

void *SaveNestedNameSpecifierAnnotation(CXXScopeSpec &SS);

void RestoreNestedNameSpecifierAnnotation(void *Annotation,
SourceRange AnnotationRange,
CXXScopeSpec &SS);

bool ShouldEnterDeclaratorScope(Scope *S, const CXXScopeSpec &SS);

bool ActOnCXXEnterDeclaratorScope(Scope *S, CXXScopeSpec &SS);

void ActOnCXXExitDeclaratorScope(Scope *S, const CXXScopeSpec &SS);

void ActOnCXXEnterDeclInitializer(Scope *S, Decl *Dcl);

void ActOnCXXExitDeclInitializer(Scope *S, Decl *Dcl);

CXXRecordDecl *createLambdaClosureType(SourceRange IntroducerRange,
TypeSourceInfo *Info,
bool KnownDependent,
LambdaCaptureDefault CaptureDefault);

CXXMethodDecl *startLambdaDefinition(CXXRecordDecl *Class,
SourceRange IntroducerRange,
TypeSourceInfo *MethodType,
SourceLocation EndLoc,
ArrayRef<ParmVarDecl *> Params,
bool IsConstexprSpecified);

void buildLambdaScope(sema::LambdaScopeInfo *LSI,
CXXMethodDecl *CallOperator,
SourceRange IntroducerRange,
LambdaCaptureDefault CaptureDefault,
SourceLocation CaptureDefaultLoc,
bool ExplicitParams,
bool ExplicitResultType,
bool Mutable);

ParsedType actOnLambdaInitCaptureInitialization(
SourceLocation Loc, bool ByRef, IdentifierInfo *Id,
LambdaCaptureInitKind InitKind, Expr *&Init) {
return ParsedType::make(buildLambdaInitCaptureInitialization(
Loc, ByRef, Id, InitKind != LambdaCaptureInitKind::CopyInit, Init));
}
QualType buildLambdaInitCaptureInitialization(SourceLocation Loc, bool ByRef,
IdentifierInfo *Id,
bool DirectInit, Expr *&Init);

VarDecl *createLambdaInitCaptureVarDecl(SourceLocation Loc,
QualType InitCaptureType,
IdentifierInfo *Id,
unsigned InitStyle, Expr *Init);

FieldDecl *buildInitCaptureField(sema::LambdaScopeInfo *LSI, VarDecl *Var);

void finishLambdaExplicitCaptures(sema::LambdaScopeInfo *LSI);

void addLambdaParameters(CXXMethodDecl *CallOperator, Scope *CurScope);

void deduceClosureReturnType(sema::CapturingScopeInfo &CSI);

void ActOnStartOfLambdaDefinition(LambdaIntroducer &Intro,
Declarator &ParamInfo, Scope *CurScope);

void ActOnLambdaError(SourceLocation StartLoc, Scope *CurScope,
bool IsInstantiation = false);

ExprResult ActOnLambdaExpr(SourceLocation StartLoc, Stmt *Body,
Scope *CurScope);

bool CaptureHasSideEffects(const sema::Capture &From);

bool DiagnoseUnusedLambdaCapture(SourceRange CaptureRange,
const sema::Capture &From);

ExprResult BuildLambdaExpr(SourceLocation StartLoc, SourceLocation EndLoc,
sema::LambdaScopeInfo *LSI);

QualType
getLambdaConversionFunctionResultType(const FunctionProtoType *CallOpType);

void DefineImplicitLambdaToFunctionPointerConversion(
SourceLocation CurrentLoc, CXXConversionDecl *Conv);

void DefineImplicitLambdaToBlockPointerConversion(SourceLocation CurrentLoc,
CXXConversionDecl *Conv);

ExprResult BuildBlockForLambdaConversion(SourceLocation CurrentLocation,
SourceLocation ConvLocation,
CXXConversionDecl *Conv,
Expr *Src);

ExprResult ParseObjCStringLiteral(SourceLocation *AtLocs,
ArrayRef<Expr *> Strings);

ExprResult BuildObjCStringLiteral(SourceLocation AtLoc, StringLiteral *S);

ExprResult BuildObjCNumericLiteral(SourceLocation AtLoc, Expr *Number);
ExprResult ActOnObjCBoolLiteral(SourceLocation AtLoc, SourceLocation ValueLoc,
bool Value);
ExprResult BuildObjCArrayLiteral(SourceRange SR, MultiExprArg Elements);

ExprResult BuildObjCBoxedExpr(SourceRange SR, Expr *ValueExpr);

ExprResult BuildObjCSubscriptExpression(SourceLocation RB, Expr *BaseExpr,
Expr *IndexExpr,
ObjCMethodDecl *getterMethod,
ObjCMethodDecl *setterMethod);

ExprResult BuildObjCDictionaryLiteral(SourceRange SR,
MutableArrayRef<ObjCDictionaryElement> Elements);

ExprResult BuildObjCEncodeExpression(SourceLocation AtLoc,
TypeSourceInfo *EncodedTypeInfo,
SourceLocation RParenLoc);
ExprResult BuildCXXMemberCallExpr(Expr *Exp, NamedDecl *FoundDecl,
CXXConversionDecl *Method,
bool HadMultipleCandidates);

ExprResult ParseObjCEncodeExpression(SourceLocation AtLoc,
SourceLocation EncodeLoc,
SourceLocation LParenLoc,
ParsedType Ty,
SourceLocation RParenLoc);

ExprResult ParseObjCSelectorExpression(Selector Sel,
SourceLocation AtLoc,
SourceLocation SelLoc,
SourceLocation LParenLoc,
SourceLocation RParenLoc,
bool WarnMultipleSelectors);

ExprResult ParseObjCProtocolExpression(IdentifierInfo * ProtocolName,
SourceLocation AtLoc,
SourceLocation ProtoLoc,
SourceLocation LParenLoc,
SourceLocation ProtoIdLoc,
SourceLocation RParenLoc);

Decl *ActOnStartLinkageSpecification(Scope *S,
SourceLocation ExternLoc,
Expr *LangStr,
SourceLocation LBraceLoc);
Decl *ActOnFinishLinkageSpecification(Scope *S,
Decl *LinkageSpec,
SourceLocation RBraceLoc);


CXXRecordDecl *getCurrentClass(Scope *S, const CXXScopeSpec *SS);
bool isCurrentClassName(const IdentifierInfo &II, Scope *S,
const CXXScopeSpec *SS = nullptr);
bool isCurrentClassNameTypo(IdentifierInfo *&II, const CXXScopeSpec *SS);

bool ActOnAccessSpecifier(AccessSpecifier Access, SourceLocation ASLoc,
SourceLocation ColonLoc,
const ParsedAttributesView &Attrs);

NamedDecl *ActOnCXXMemberDeclarator(Scope *S, AccessSpecifier AS,
Declarator &D,
MultiTemplateParamsArg TemplateParameterLists,
Expr *BitfieldWidth, const VirtSpecifiers &VS,
InClassInitStyle InitStyle);

void ActOnStartCXXInClassMemberInitializer();
void ActOnFinishCXXInClassMemberInitializer(Decl *VarDecl,
SourceLocation EqualLoc,
Expr *Init);

MemInitResult ActOnMemInitializer(Decl *ConstructorD,
Scope *S,
CXXScopeSpec &SS,
IdentifierInfo *MemberOrBase,
ParsedType TemplateTypeTy,
const DeclSpec &DS,
SourceLocation IdLoc,
SourceLocation LParenLoc,
ArrayRef<Expr *> Args,
SourceLocation RParenLoc,
SourceLocation EllipsisLoc);

MemInitResult ActOnMemInitializer(Decl *ConstructorD,
Scope *S,
CXXScopeSpec &SS,
IdentifierInfo *MemberOrBase,
ParsedType TemplateTypeTy,
const DeclSpec &DS,
SourceLocation IdLoc,
Expr *InitList,
SourceLocation EllipsisLoc);

MemInitResult BuildMemInitializer(Decl *ConstructorD,
Scope *S,
CXXScopeSpec &SS,
IdentifierInfo *MemberOrBase,
ParsedType TemplateTypeTy,
const DeclSpec &DS,
SourceLocation IdLoc,
Expr *Init,
SourceLocation EllipsisLoc);

MemInitResult BuildMemberInitializer(ValueDecl *Member,
Expr *Init,
SourceLocation IdLoc);

MemInitResult BuildBaseInitializer(QualType BaseType,
TypeSourceInfo *BaseTInfo,
Expr *Init,
CXXRecordDecl *ClassDecl,
SourceLocation EllipsisLoc);

MemInitResult BuildDelegatingInitializer(TypeSourceInfo *TInfo,
Expr *Init,
CXXRecordDecl *ClassDecl);

bool SetDelegatingInitializer(CXXConstructorDecl *Constructor,
CXXCtorInitializer *Initializer);

bool SetCtorInitializers(CXXConstructorDecl *Constructor, bool AnyErrors,
ArrayRef<CXXCtorInitializer *> Initializers = None);

void SetIvarInitializers(ObjCImplementationDecl *ObjCImplementation);


void MarkBaseAndMemberDestructorsReferenced(SourceLocation Loc,
CXXRecordDecl *Record);

typedef std::pair<CXXRecordDecl*, SourceLocation> VTableUse;

SmallVector<VTableUse, 16> VTableUses;

llvm::DenseMap<CXXRecordDecl *, bool> VTablesUsed;

void LoadExternalVTableUses();

void MarkVTableUsed(SourceLocation Loc, CXXRecordDecl *Class,
bool DefinitionRequired = false);

void MarkVirtualMemberExceptionSpecsNeeded(SourceLocation Loc,
const CXXRecordDecl *RD);

void MarkVirtualMembersReferenced(SourceLocation Loc,
const CXXRecordDecl *RD);

bool DefineUsedVTables();

void AddImplicitlyDeclaredMembersToClass(CXXRecordDecl *ClassDecl);

void ActOnMemInitializers(Decl *ConstructorDecl,
SourceLocation ColonLoc,
ArrayRef<CXXCtorInitializer*> MemInits,
bool AnyErrors);

void checkClassLevelDLLAttribute(CXXRecordDecl *Class);
void checkClassLevelCodeSegAttribute(CXXRecordDecl *Class);

void referenceDLLExportedClassMethods();

void propagateDLLAttrToBaseClassTemplate(
CXXRecordDecl *Class, Attr *ClassAttr,
ClassTemplateSpecializationDecl *BaseTemplateSpec,
SourceLocation BaseLoc);

void CheckCompletedCXXClass(CXXRecordDecl *Record);

void checkIllFormedTrivialABIStruct(CXXRecordDecl &RD);

void ActOnFinishCXXMemberSpecification(Scope *S, SourceLocation RLoc,
Decl *TagDecl, SourceLocation LBrac,
SourceLocation RBrac,
const ParsedAttributesView &AttrList);
void ActOnFinishCXXMemberDecls();
void ActOnFinishCXXNonNestedClass(Decl *D);

void ActOnReenterCXXMethodParameter(Scope *S, ParmVarDecl *Param);
unsigned ActOnReenterTemplateScope(Scope *S, Decl *Template);
void ActOnStartDelayedMemberDeclarations(Scope *S, Decl *Record);
void ActOnStartDelayedCXXMethodDeclaration(Scope *S, Decl *Method);
void ActOnDelayedCXXMethodParameter(Scope *S, Decl *Param);
void ActOnFinishDelayedMemberDeclarations(Scope *S, Decl *Record);
void ActOnFinishDelayedCXXMethodDeclaration(Scope *S, Decl *Method);
void ActOnFinishDelayedMemberInitializers(Decl *Record);
void MarkAsLateParsedTemplate(FunctionDecl *FD, Decl *FnD,
CachedTokens &Toks);
void UnmarkAsLateParsedTemplate(FunctionDecl *FD);
bool IsInsideALocalClassWithinATemplateFunction();

Decl *ActOnStaticAssertDeclaration(SourceLocation StaticAssertLoc,
Expr *AssertExpr,
Expr *AssertMessageExpr,
SourceLocation RParenLoc);
Decl *BuildStaticAssertDeclaration(SourceLocation StaticAssertLoc,
Expr *AssertExpr,
StringLiteral *AssertMessageExpr,
SourceLocation RParenLoc,
bool Failed);

FriendDecl *CheckFriendTypeDecl(SourceLocation LocStart,
SourceLocation FriendLoc,
TypeSourceInfo *TSInfo);
Decl *ActOnFriendTypeDecl(Scope *S, const DeclSpec &DS,
MultiTemplateParamsArg TemplateParams);
NamedDecl *ActOnFriendFunctionDecl(Scope *S, Declarator &D,
MultiTemplateParamsArg TemplateParams);

QualType CheckConstructorDeclarator(Declarator &D, QualType R,
StorageClass& SC);
void CheckConstructor(CXXConstructorDecl *Constructor);
QualType CheckDestructorDeclarator(Declarator &D, QualType R,
StorageClass& SC);
bool CheckDestructor(CXXDestructorDecl *Destructor);
void CheckConversionDeclarator(Declarator &D, QualType &R,
StorageClass& SC);
Decl *ActOnConversionDeclarator(CXXConversionDecl *Conversion);
void CheckDeductionGuideDeclarator(Declarator &D, QualType &R,
StorageClass &SC);
void CheckDeductionGuideTemplate(FunctionTemplateDecl *TD);

void CheckExplicitlyDefaultedSpecialMember(CXXMethodDecl *MD);
void CheckExplicitlyDefaultedMemberExceptionSpec(CXXMethodDecl *MD,
const FunctionProtoType *T);
void CheckDelayedMemberExceptionSpecs();


CXXBaseSpecifier *CheckBaseSpecifier(CXXRecordDecl *Class,
SourceRange SpecifierRange,
bool Virtual, AccessSpecifier Access,
TypeSourceInfo *TInfo,
SourceLocation EllipsisLoc);

BaseResult ActOnBaseSpecifier(Decl *classdecl,
SourceRange SpecifierRange,
ParsedAttributes &Attrs,
bool Virtual, AccessSpecifier Access,
ParsedType basetype,
SourceLocation BaseLoc,
SourceLocation EllipsisLoc);

bool AttachBaseSpecifiers(CXXRecordDecl *Class,
MutableArrayRef<CXXBaseSpecifier *> Bases);
void ActOnBaseSpecifiers(Decl *ClassDecl,
MutableArrayRef<CXXBaseSpecifier *> Bases);

bool IsDerivedFrom(SourceLocation Loc, QualType Derived, QualType Base);
bool IsDerivedFrom(SourceLocation Loc, QualType Derived, QualType Base,
CXXBasePaths &Paths);

void BuildBasePathArray(const CXXBasePaths &Paths, CXXCastPath &BasePath);

bool CheckDerivedToBaseConversion(QualType Derived, QualType Base,
SourceLocation Loc, SourceRange Range,
CXXCastPath *BasePath = nullptr,
bool IgnoreAccess = false);
bool CheckDerivedToBaseConversion(QualType Derived, QualType Base,
unsigned InaccessibleBaseID,
unsigned AmbigiousBaseConvID,
SourceLocation Loc, SourceRange Range,
DeclarationName Name,
CXXCastPath *BasePath,
bool IgnoreAccess = false);

std::string getAmbiguousPathsDisplayString(CXXBasePaths &Paths);

bool CheckOverridingFunctionAttributes(const CXXMethodDecl *New,
const CXXMethodDecl *Old);

bool CheckOverridingFunctionReturnType(const CXXMethodDecl *New,
const CXXMethodDecl *Old);

bool CheckOverridingFunctionExceptionSpec(const CXXMethodDecl *New,
const CXXMethodDecl *Old);

bool CheckPureMethod(CXXMethodDecl *Method, SourceRange InitRange);

void CheckOverrideControl(NamedDecl *D);

void DiagnoseAbsenceOfOverrideControl(NamedDecl *D);

bool CheckIfOverriddenFunctionIsMarkedFinal(const CXXMethodDecl *New,
const CXXMethodDecl *Old);



enum AccessResult {
AR_accessible,
AR_inaccessible,
AR_dependent,
AR_delayed
};

bool SetMemberAccessSpecifier(NamedDecl *MemberDecl,
NamedDecl *PrevMemberDecl,
AccessSpecifier LexicalAS);

AccessResult CheckUnresolvedMemberAccess(UnresolvedMemberExpr *E,
DeclAccessPair FoundDecl);
AccessResult CheckUnresolvedLookupAccess(UnresolvedLookupExpr *E,
DeclAccessPair FoundDecl);
AccessResult CheckAllocationAccess(SourceLocation OperatorLoc,
SourceRange PlacementRange,
CXXRecordDecl *NamingClass,
DeclAccessPair FoundDecl,
bool Diagnose = true);
AccessResult CheckConstructorAccess(SourceLocation Loc,
CXXConstructorDecl *D,
DeclAccessPair FoundDecl,
const InitializedEntity &Entity,
bool IsCopyBindingRefToTemp = false);
AccessResult CheckConstructorAccess(SourceLocation Loc,
CXXConstructorDecl *D,
DeclAccessPair FoundDecl,
const InitializedEntity &Entity,
const PartialDiagnostic &PDiag);
AccessResult CheckDestructorAccess(SourceLocation Loc,
CXXDestructorDecl *Dtor,
const PartialDiagnostic &PDiag,
QualType objectType = QualType());
AccessResult CheckFriendAccess(NamedDecl *D);
AccessResult CheckMemberAccess(SourceLocation UseLoc,
CXXRecordDecl *NamingClass,
DeclAccessPair Found);
AccessResult CheckMemberOperatorAccess(SourceLocation Loc,
Expr *ObjectExpr,
Expr *ArgExpr,
DeclAccessPair FoundDecl);
AccessResult CheckAddressOfMemberAccess(Expr *OvlExpr,
DeclAccessPair FoundDecl);
AccessResult CheckBaseClassAccess(SourceLocation AccessLoc,
QualType Base, QualType Derived,
const CXXBasePath &Path,
unsigned DiagID,
bool ForceCheck = false,
bool ForceUnprivileged = false);
void CheckLookupAccess(const LookupResult &R);
bool IsSimplyAccessible(NamedDecl *decl, DeclContext *Ctx);
bool isSpecialMemberAccessibleForDeletion(CXXMethodDecl *decl,
AccessSpecifier access,
QualType objectType);

void HandleDependentAccessCheck(const DependentDiagnostic &DD,
const MultiLevelTemplateArgumentList &TemplateArgs);
void PerformDependentDiagnostics(const DeclContext *Pattern,
const MultiLevelTemplateArgumentList &TemplateArgs);

void HandleDelayedAccessCheck(sema::DelayedDiagnostic &DD, Decl *Ctx);

bool AccessCheckingSFINAE;

enum AbstractDiagSelID {
AbstractNone = -1,
AbstractReturnType,
AbstractParamType,
AbstractVariableType,
AbstractFieldType,
AbstractIvarType,
AbstractSynthesizedIvarType,
AbstractArrayType
};

bool isAbstractType(SourceLocation Loc, QualType T);
bool RequireNonAbstractType(SourceLocation Loc, QualType T,
TypeDiagnoser &Diagnoser);
template <typename... Ts>
bool RequireNonAbstractType(SourceLocation Loc, QualType T, unsigned DiagID,
const Ts &...Args) {
BoundTypeDiagnoser<Ts...> Diagnoser(DiagID, Args...);
return RequireNonAbstractType(Loc, T, Diagnoser);
}

void DiagnoseAbstractType(const CXXRecordDecl *RD);


bool CheckOverloadedOperatorDeclaration(FunctionDecl *FnDecl);

bool CheckLiteralOperatorDeclaration(FunctionDecl *FnDecl);

void FilterAcceptableTemplateNames(LookupResult &R,
bool AllowFunctionTemplates = true);
bool hasAnyAcceptableTemplateNames(LookupResult &R,
bool AllowFunctionTemplates = true);

bool LookupTemplateName(LookupResult &R, Scope *S, CXXScopeSpec &SS,
QualType ObjectType, bool EnteringContext,
bool &MemberOfUnknownSpecialization,
SourceLocation TemplateKWLoc = SourceLocation());

TemplateNameKind isTemplateName(Scope *S,
CXXScopeSpec &SS,
bool hasTemplateKeyword,
const UnqualifiedId &Name,
ParsedType ObjectType,
bool EnteringContext,
TemplateTy &Template,
bool &MemberOfUnknownSpecialization);

bool isDeductionGuideName(Scope *S, const IdentifierInfo &Name,
SourceLocation NameLoc,
ParsedTemplateTy *Template = nullptr);

bool DiagnoseUnknownTemplateName(const IdentifierInfo &II,
SourceLocation IILoc,
Scope *S,
const CXXScopeSpec *SS,
TemplateTy &SuggestedTemplate,
TemplateNameKind &SuggestedKind);

bool DiagnoseUninstantiableTemplate(SourceLocation PointOfInstantiation,
NamedDecl *Instantiation,
bool InstantiatedFromMember,
const NamedDecl *Pattern,
const NamedDecl *PatternDef,
TemplateSpecializationKind TSK,
bool Complain = true);

void DiagnoseTemplateParameterShadow(SourceLocation Loc, Decl *PrevDecl);
TemplateDecl *AdjustDeclIfTemplate(Decl *&Decl);

NamedDecl *ActOnTypeParameter(Scope *S, bool Typename,
SourceLocation EllipsisLoc,
SourceLocation KeyLoc,
IdentifierInfo *ParamName,
SourceLocation ParamNameLoc,
unsigned Depth, unsigned Position,
SourceLocation EqualLoc,
ParsedType DefaultArg);

QualType CheckNonTypeTemplateParameterType(TypeSourceInfo *&TSI,
SourceLocation Loc);
QualType CheckNonTypeTemplateParameterType(QualType T, SourceLocation Loc);

NamedDecl *ActOnNonTypeTemplateParameter(Scope *S, Declarator &D,
unsigned Depth,
unsigned Position,
SourceLocation EqualLoc,
Expr *DefaultArg);
NamedDecl *ActOnTemplateTemplateParameter(Scope *S,
SourceLocation TmpLoc,
TemplateParameterList *Params,
SourceLocation EllipsisLoc,
IdentifierInfo *ParamName,
SourceLocation ParamNameLoc,
unsigned Depth,
unsigned Position,
SourceLocation EqualLoc,
ParsedTemplateArgument DefaultArg);

TemplateParameterList *
ActOnTemplateParameterList(unsigned Depth,
SourceLocation ExportLoc,
SourceLocation TemplateLoc,
SourceLocation LAngleLoc,
ArrayRef<NamedDecl *> Params,
SourceLocation RAngleLoc,
Expr *RequiresClause);

enum TemplateParamListContext {
TPC_ClassTemplate,
TPC_VarTemplate,
TPC_FunctionTemplate,
TPC_ClassTemplateMember,
TPC_FriendClassTemplate,
TPC_FriendFunctionTemplate,
TPC_FriendFunctionTemplateDefinition,
TPC_TypeAliasTemplate
};

bool CheckTemplateParameterList(TemplateParameterList *NewParams,
TemplateParameterList *OldParams,
TemplateParamListContext TPC);
TemplateParameterList *MatchTemplateParametersToScopeSpecifier(
SourceLocation DeclStartLoc, SourceLocation DeclLoc,
const CXXScopeSpec &SS, TemplateIdAnnotation *TemplateId,
ArrayRef<TemplateParameterList *> ParamLists,
bool IsFriend, bool &IsMemberSpecialization, bool &Invalid);

DeclResult CheckClassTemplate(
Scope *S, unsigned TagSpec, TagUseKind TUK, SourceLocation KWLoc,
CXXScopeSpec &SS, IdentifierInfo *Name, SourceLocation NameLoc,
const ParsedAttributesView &Attr, TemplateParameterList *TemplateParams,
AccessSpecifier AS, SourceLocation ModulePrivateLoc,
SourceLocation FriendLoc, unsigned NumOuterTemplateParamLists,
TemplateParameterList **OuterTemplateParamLists,
SkipBodyInfo *SkipBody = nullptr);

TemplateArgumentLoc getTrivialTemplateArgumentLoc(const TemplateArgument &Arg,
QualType NTTPType,
SourceLocation Loc);

void translateTemplateArguments(const ASTTemplateArgsPtr &In,
TemplateArgumentListInfo &Out);

ParsedTemplateArgument ActOnTemplateTypeArgument(TypeResult ParsedType);

void NoteAllFoundTemplates(TemplateName Name);

QualType CheckTemplateIdType(TemplateName Template,
SourceLocation TemplateLoc,
TemplateArgumentListInfo &TemplateArgs);

TypeResult
ActOnTemplateIdType(CXXScopeSpec &SS, SourceLocation TemplateKWLoc,
TemplateTy Template, IdentifierInfo *TemplateII,
SourceLocation TemplateIILoc,
SourceLocation LAngleLoc,
ASTTemplateArgsPtr TemplateArgs,
SourceLocation RAngleLoc,
bool IsCtorOrDtorName = false,
bool IsClassName = false);

TypeResult ActOnTagTemplateIdType(TagUseKind TUK,
TypeSpecifierType TagSpec,
SourceLocation TagLoc,
CXXScopeSpec &SS,
SourceLocation TemplateKWLoc,
TemplateTy TemplateD,
SourceLocation TemplateLoc,
SourceLocation LAngleLoc,
ASTTemplateArgsPtr TemplateArgsIn,
SourceLocation RAngleLoc);

DeclResult ActOnVarTemplateSpecialization(
Scope *S, Declarator &D, TypeSourceInfo *DI,
SourceLocation TemplateKWLoc, TemplateParameterList *TemplateParams,
StorageClass SC, bool IsPartialSpecialization);

DeclResult CheckVarTemplateId(VarTemplateDecl *Template,
SourceLocation TemplateLoc,
SourceLocation TemplateNameLoc,
const TemplateArgumentListInfo &TemplateArgs);

ExprResult CheckVarTemplateId(const CXXScopeSpec &SS,
const DeclarationNameInfo &NameInfo,
VarTemplateDecl *Template,
SourceLocation TemplateLoc,
const TemplateArgumentListInfo *TemplateArgs);

void diagnoseMissingTemplateArguments(TemplateName Name, SourceLocation Loc);

ExprResult BuildTemplateIdExpr(const CXXScopeSpec &SS,
SourceLocation TemplateKWLoc,
LookupResult &R,
bool RequiresADL,
const TemplateArgumentListInfo *TemplateArgs);

ExprResult BuildQualifiedTemplateIdExpr(CXXScopeSpec &SS,
SourceLocation TemplateKWLoc,
const DeclarationNameInfo &NameInfo,
const TemplateArgumentListInfo *TemplateArgs);

TemplateNameKind ActOnDependentTemplateName(
Scope *S, CXXScopeSpec &SS, SourceLocation TemplateKWLoc,
const UnqualifiedId &Name, ParsedType ObjectType, bool EnteringContext,
TemplateTy &Template, bool AllowInjectedClassName = false);

DeclResult ActOnClassTemplateSpecialization(
Scope *S, unsigned TagSpec, TagUseKind TUK, SourceLocation KWLoc,
SourceLocation ModulePrivateLoc, TemplateIdAnnotation &TemplateId,
const ParsedAttributesView &Attr,
MultiTemplateParamsArg TemplateParameterLists,
SkipBodyInfo *SkipBody = nullptr);

bool CheckTemplatePartialSpecializationArgs(SourceLocation Loc,
TemplateDecl *PrimaryTemplate,
unsigned NumExplicitArgs,
ArrayRef<TemplateArgument> Args);
void CheckTemplatePartialSpecialization(
ClassTemplatePartialSpecializationDecl *Partial);
void CheckTemplatePartialSpecialization(
VarTemplatePartialSpecializationDecl *Partial);

Decl *ActOnTemplateDeclarator(Scope *S,
MultiTemplateParamsArg TemplateParameterLists,
Declarator &D);

bool
CheckSpecializationInstantiationRedecl(SourceLocation NewLoc,
TemplateSpecializationKind NewTSK,
NamedDecl *PrevDecl,
TemplateSpecializationKind PrevTSK,
SourceLocation PrevPtOfInstantiation,
bool &SuppressNew);

bool CheckDependentFunctionTemplateSpecialization(FunctionDecl *FD,
const TemplateArgumentListInfo &ExplicitTemplateArgs,
LookupResult &Previous);

bool CheckFunctionTemplateSpecialization(FunctionDecl *FD,
TemplateArgumentListInfo *ExplicitTemplateArgs,
LookupResult &Previous);
bool CheckMemberSpecialization(NamedDecl *Member, LookupResult &Previous);
void CompleteMemberSpecialization(NamedDecl *Member, LookupResult &Previous);

DeclResult ActOnExplicitInstantiation(
Scope *S, SourceLocation ExternLoc, SourceLocation TemplateLoc,
unsigned TagSpec, SourceLocation KWLoc, const CXXScopeSpec &SS,
TemplateTy Template, SourceLocation TemplateNameLoc,
SourceLocation LAngleLoc, ASTTemplateArgsPtr TemplateArgs,
SourceLocation RAngleLoc, const ParsedAttributesView &Attr);

DeclResult ActOnExplicitInstantiation(Scope *S, SourceLocation ExternLoc,
SourceLocation TemplateLoc,
unsigned TagSpec, SourceLocation KWLoc,
CXXScopeSpec &SS, IdentifierInfo *Name,
SourceLocation NameLoc,
const ParsedAttributesView &Attr);

DeclResult ActOnExplicitInstantiation(Scope *S,
SourceLocation ExternLoc,
SourceLocation TemplateLoc,
Declarator &D);

TemplateArgumentLoc
SubstDefaultTemplateArgumentIfAvailable(TemplateDecl *Template,
SourceLocation TemplateLoc,
SourceLocation RAngleLoc,
Decl *Param,
SmallVectorImpl<TemplateArgument>
&Converted,
bool &HasDefaultArg);

enum CheckTemplateArgumentKind {
CTAK_Specified,

CTAK_Deduced,

CTAK_DeducedFromArrayBound
};

bool CheckTemplateArgument(NamedDecl *Param,
TemplateArgumentLoc &Arg,
NamedDecl *Template,
SourceLocation TemplateLoc,
SourceLocation RAngleLoc,
unsigned ArgumentPackIndex,
SmallVectorImpl<TemplateArgument> &Converted,
CheckTemplateArgumentKind CTAK = CTAK_Specified);

bool CheckTemplateArgumentList(TemplateDecl *Template,
SourceLocation TemplateLoc,
TemplateArgumentListInfo &TemplateArgs,
bool PartialTemplateArgs,
SmallVectorImpl<TemplateArgument> &Converted,
bool UpdateArgsWithConversions = true);

bool CheckTemplateTypeArgument(TemplateTypeParmDecl *Param,
TemplateArgumentLoc &Arg,
SmallVectorImpl<TemplateArgument> &Converted);

bool CheckTemplateArgument(TemplateTypeParmDecl *Param,
TypeSourceInfo *Arg);
ExprResult CheckTemplateArgument(NonTypeTemplateParmDecl *Param,
QualType InstantiatedParamType, Expr *Arg,
TemplateArgument &Converted,
CheckTemplateArgumentKind CTAK = CTAK_Specified);
bool CheckTemplateTemplateArgument(TemplateParameterList *Params,
TemplateArgumentLoc &Arg);

ExprResult
BuildExpressionFromDeclTemplateArgument(const TemplateArgument &Arg,
QualType ParamType,
SourceLocation Loc);
ExprResult
BuildExpressionFromIntegralTemplateArgument(const TemplateArgument &Arg,
SourceLocation Loc);

enum TemplateParameterListEqualKind {
TPL_TemplateMatch,

TPL_TemplateTemplateParmMatch,

TPL_TemplateTemplateArgumentMatch
};

bool TemplateParameterListsAreEqual(TemplateParameterList *New,
TemplateParameterList *Old,
bool Complain,
TemplateParameterListEqualKind Kind,
SourceLocation TemplateArgLoc
= SourceLocation());

bool CheckTemplateDeclScope(Scope *S, TemplateParameterList *TemplateParams);

TypeResult
ActOnTypenameType(Scope *S, SourceLocation TypenameLoc,
const CXXScopeSpec &SS, const IdentifierInfo &II,
SourceLocation IdLoc);

TypeResult
ActOnTypenameType(Scope *S, SourceLocation TypenameLoc,
const CXXScopeSpec &SS,
SourceLocation TemplateLoc,
TemplateTy TemplateName,
IdentifierInfo *TemplateII,
SourceLocation TemplateIILoc,
SourceLocation LAngleLoc,
ASTTemplateArgsPtr TemplateArgs,
SourceLocation RAngleLoc);

QualType CheckTypenameType(ElaboratedTypeKeyword Keyword,
SourceLocation KeywordLoc,
NestedNameSpecifierLoc QualifierLoc,
const IdentifierInfo &II,
SourceLocation IILoc);

TypeSourceInfo *RebuildTypeInCurrentInstantiation(TypeSourceInfo *T,
SourceLocation Loc,
DeclarationName Name);
bool RebuildNestedNameSpecifierInCurrentInstantiation(CXXScopeSpec &SS);

ExprResult RebuildExprInCurrentInstantiation(Expr *E);
bool RebuildTemplateParamsInCurrentInstantiation(
TemplateParameterList *Params);

std::string
getTemplateArgumentBindingsText(const TemplateParameterList *Params,
const TemplateArgumentList &Args);

std::string
getTemplateArgumentBindingsText(const TemplateParameterList *Params,
const TemplateArgument *Args,
unsigned NumArgs);


bool isUnexpandedParameterPackPermitted();

enum UnexpandedParameterPackContext {
UPPC_Expression = 0,

UPPC_BaseType,

UPPC_DeclarationType,

UPPC_DataMemberType,

UPPC_BitFieldWidth,

UPPC_StaticAssertExpression,

UPPC_FixedUnderlyingType,

UPPC_EnumeratorValue,

UPPC_UsingDeclaration,

UPPC_FriendDeclaration,

UPPC_DeclarationQualifier,

UPPC_Initializer,

UPPC_DefaultArgument,

UPPC_NonTypeTemplateParameterType,

UPPC_ExceptionType,

UPPC_PartialSpecialization,

UPPC_IfExists,

UPPC_IfNotExists,

UPPC_Lambda,

UPPC_Block
};

bool DiagnoseUnexpandedParameterPacks(SourceLocation Loc,
UnexpandedParameterPackContext UPPC,
ArrayRef<UnexpandedParameterPack> Unexpanded);

bool DiagnoseUnexpandedParameterPack(SourceLocation Loc, TypeSourceInfo *T,
UnexpandedParameterPackContext UPPC);

bool DiagnoseUnexpandedParameterPack(Expr *E,
UnexpandedParameterPackContext UPPC = UPPC_Expression);

bool DiagnoseUnexpandedParameterPack(const CXXScopeSpec &SS,
UnexpandedParameterPackContext UPPC);

bool DiagnoseUnexpandedParameterPack(const DeclarationNameInfo &NameInfo,
UnexpandedParameterPackContext UPPC);

bool DiagnoseUnexpandedParameterPack(SourceLocation Loc,
TemplateName Template,
UnexpandedParameterPackContext UPPC);

bool DiagnoseUnexpandedParameterPack(TemplateArgumentLoc Arg,
UnexpandedParameterPackContext UPPC);

void collectUnexpandedParameterPacks(TemplateArgument Arg,
SmallVectorImpl<UnexpandedParameterPack> &Unexpanded);

void collectUnexpandedParameterPacks(TemplateArgumentLoc Arg,
SmallVectorImpl<UnexpandedParameterPack> &Unexpanded);

void collectUnexpandedParameterPacks(QualType T,
SmallVectorImpl<UnexpandedParameterPack> &Unexpanded);

void collectUnexpandedParameterPacks(TypeLoc TL,
SmallVectorImpl<UnexpandedParameterPack> &Unexpanded);

void collectUnexpandedParameterPacks(NestedNameSpecifierLoc NNS,
SmallVectorImpl<UnexpandedParameterPack> &Unexpanded);

void collectUnexpandedParameterPacks(const DeclarationNameInfo &NameInfo,
SmallVectorImpl<UnexpandedParameterPack> &Unexpanded);

ParsedTemplateArgument ActOnPackExpansion(const ParsedTemplateArgument &Arg,
SourceLocation EllipsisLoc);

TypeResult ActOnPackExpansion(ParsedType Type, SourceLocation EllipsisLoc);

TypeSourceInfo *CheckPackExpansion(TypeSourceInfo *Pattern,
SourceLocation EllipsisLoc,
Optional<unsigned> NumExpansions);

QualType CheckPackExpansion(QualType Pattern,
SourceRange PatternRange,
SourceLocation EllipsisLoc,
Optional<unsigned> NumExpansions);

ExprResult ActOnPackExpansion(Expr *Pattern, SourceLocation EllipsisLoc);

ExprResult CheckPackExpansion(Expr *Pattern, SourceLocation EllipsisLoc,
Optional<unsigned> NumExpansions);

bool CheckParameterPacksForExpansion(SourceLocation EllipsisLoc,
SourceRange PatternRange,
ArrayRef<UnexpandedParameterPack> Unexpanded,
const MultiLevelTemplateArgumentList &TemplateArgs,
bool &ShouldExpand,
bool &RetainExpansion,
Optional<unsigned> &NumExpansions);

Optional<unsigned> getNumArgumentsInExpansion(QualType T,
const MultiLevelTemplateArgumentList &TemplateArgs);

bool containsUnexpandedParameterPacks(Declarator &D);

TemplateArgumentLoc getTemplateArgumentPackExpansionPattern(
TemplateArgumentLoc OrigLoc,
SourceLocation &Ellipsis,
Optional<unsigned> &NumExpansions) const;

Optional<unsigned> getFullyPackExpandedSize(TemplateArgument Arg);


QualType adjustCCAndNoReturn(QualType ArgFunctionType, QualType FunctionType,
bool AdjustExceptionSpec = false);

enum TemplateDeductionResult {
TDK_Success = 0,
TDK_Invalid,
TDK_InstantiationDepth,
TDK_Incomplete,
TDK_IncompletePack,
TDK_Inconsistent,
TDK_Underqualified,
TDK_SubstitutionFailure,
TDK_DeducedMismatch,
TDK_DeducedMismatchNested,
TDK_NonDeducedMismatch,
TDK_TooManyArguments,
TDK_TooFewArguments,
TDK_InvalidExplicitArguments,
TDK_NonDependentConversionFailure,
TDK_MiscellaneousDeductionFailure,
TDK_CUDATargetMismatch
};

TemplateDeductionResult
DeduceTemplateArguments(ClassTemplatePartialSpecializationDecl *Partial,
const TemplateArgumentList &TemplateArgs,
sema::TemplateDeductionInfo &Info);

TemplateDeductionResult
DeduceTemplateArguments(VarTemplatePartialSpecializationDecl *Partial,
const TemplateArgumentList &TemplateArgs,
sema::TemplateDeductionInfo &Info);

TemplateDeductionResult SubstituteExplicitTemplateArguments(
FunctionTemplateDecl *FunctionTemplate,
TemplateArgumentListInfo &ExplicitTemplateArgs,
SmallVectorImpl<DeducedTemplateArgument> &Deduced,
SmallVectorImpl<QualType> &ParamTypes, QualType *FunctionType,
sema::TemplateDeductionInfo &Info);

struct OriginalCallArg {
OriginalCallArg(QualType OriginalParamType, bool DecomposedParam,
unsigned ArgIdx, QualType OriginalArgType)
: OriginalParamType(OriginalParamType),
DecomposedParam(DecomposedParam), ArgIdx(ArgIdx),
OriginalArgType(OriginalArgType) {}

QualType OriginalParamType;
bool DecomposedParam;
unsigned ArgIdx;
QualType OriginalArgType;
};

TemplateDeductionResult FinishTemplateArgumentDeduction(
FunctionTemplateDecl *FunctionTemplate,
SmallVectorImpl<DeducedTemplateArgument> &Deduced,
unsigned NumExplicitlySpecified, FunctionDecl *&Specialization,
sema::TemplateDeductionInfo &Info,
SmallVectorImpl<OriginalCallArg> const *OriginalCallArgs = nullptr,
bool PartialOverloading = false,
llvm::function_ref<bool()> CheckNonDependent = []{ return false; });

TemplateDeductionResult DeduceTemplateArguments(
FunctionTemplateDecl *FunctionTemplate,
TemplateArgumentListInfo *ExplicitTemplateArgs, ArrayRef<Expr *> Args,
FunctionDecl *&Specialization, sema::TemplateDeductionInfo &Info,
bool PartialOverloading,
llvm::function_ref<bool(ArrayRef<QualType>)> CheckNonDependent);

TemplateDeductionResult
DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
TemplateArgumentListInfo *ExplicitTemplateArgs,
QualType ArgFunctionType,
FunctionDecl *&Specialization,
sema::TemplateDeductionInfo &Info,
bool IsAddressOfFunction = false);

TemplateDeductionResult
DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
QualType ToType,
CXXConversionDecl *&Specialization,
sema::TemplateDeductionInfo &Info);

TemplateDeductionResult
DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
TemplateArgumentListInfo *ExplicitTemplateArgs,
FunctionDecl *&Specialization,
sema::TemplateDeductionInfo &Info,
bool IsAddressOfFunction = false);

QualType SubstAutoType(QualType TypeWithAuto, QualType Replacement);
TypeSourceInfo* SubstAutoTypeSourceInfo(TypeSourceInfo *TypeWithAuto,
QualType Replacement);
QualType ReplaceAutoType(QualType TypeWithAuto, QualType Replacement);

enum DeduceAutoResult {
DAR_Succeeded,
DAR_Failed,
DAR_FailedAlreadyDiagnosed
};

DeduceAutoResult
DeduceAutoType(TypeSourceInfo *AutoType, Expr *&Initializer, QualType &Result,
Optional<unsigned> DependentDeductionDepth = None);
DeduceAutoResult
DeduceAutoType(TypeLoc AutoTypeLoc, Expr *&Initializer, QualType &Result,
Optional<unsigned> DependentDeductionDepth = None);
void DiagnoseAutoDeductionFailure(VarDecl *VDecl, Expr *Init);
bool DeduceReturnType(FunctionDecl *FD, SourceLocation Loc,
bool Diagnose = true);

void DeclareImplicitDeductionGuides(TemplateDecl *Template,
SourceLocation Loc);

QualType DeduceTemplateSpecializationFromInitializer(
TypeSourceInfo *TInfo, const InitializedEntity &Entity,
const InitializationKind &Kind, MultiExprArg Init);

QualType deduceVarTypeFromInitializer(VarDecl *VDecl, DeclarationName Name,
QualType Type, TypeSourceInfo *TSI,
SourceRange Range, bool DirectInit,
Expr *Init);

TypeLoc getReturnTypeLoc(FunctionDecl *FD) const;

bool DeduceFunctionTypeFromReturnExpr(FunctionDecl *FD,
SourceLocation ReturnLoc,
Expr *&RetExpr, AutoType *AT);

FunctionTemplateDecl *getMoreSpecializedTemplate(FunctionTemplateDecl *FT1,
FunctionTemplateDecl *FT2,
SourceLocation Loc,
TemplatePartialOrderingContext TPOC,
unsigned NumCallArguments1,
unsigned NumCallArguments2);
UnresolvedSetIterator
getMostSpecialized(UnresolvedSetIterator SBegin, UnresolvedSetIterator SEnd,
TemplateSpecCandidateSet &FailedCandidates,
SourceLocation Loc,
const PartialDiagnostic &NoneDiag,
const PartialDiagnostic &AmbigDiag,
const PartialDiagnostic &CandidateDiag,
bool Complain = true, QualType TargetType = QualType());

ClassTemplatePartialSpecializationDecl *
getMoreSpecializedPartialSpecialization(
ClassTemplatePartialSpecializationDecl *PS1,
ClassTemplatePartialSpecializationDecl *PS2,
SourceLocation Loc);

bool isMoreSpecializedThanPrimary(ClassTemplatePartialSpecializationDecl *T,
sema::TemplateDeductionInfo &Info);

VarTemplatePartialSpecializationDecl *getMoreSpecializedPartialSpecialization(
VarTemplatePartialSpecializationDecl *PS1,
VarTemplatePartialSpecializationDecl *PS2, SourceLocation Loc);

bool isMoreSpecializedThanPrimary(VarTemplatePartialSpecializationDecl *T,
sema::TemplateDeductionInfo &Info);

bool isTemplateTemplateParameterAtLeastAsSpecializedAs(
TemplateParameterList *P, TemplateDecl *AArg, SourceLocation Loc);

void MarkUsedTemplateParameters(const TemplateArgumentList &TemplateArgs,
bool OnlyDeduced,
unsigned Depth,
llvm::SmallBitVector &Used);
void MarkDeducedTemplateParameters(
const FunctionTemplateDecl *FunctionTemplate,
llvm::SmallBitVector &Deduced) {
return MarkDeducedTemplateParameters(Context, FunctionTemplate, Deduced);
}
static void MarkDeducedTemplateParameters(ASTContext &Ctx,
const FunctionTemplateDecl *FunctionTemplate,
llvm::SmallBitVector &Deduced);


MultiLevelTemplateArgumentList
getTemplateInstantiationArgs(NamedDecl *D,
const TemplateArgumentList *Innermost = nullptr,
bool RelativeToPrimary = false,
const FunctionDecl *Pattern = nullptr);

struct CodeSynthesisContext {
enum SynthesisKind {
TemplateInstantiation,

DefaultTemplateArgumentInstantiation,

DefaultFunctionArgumentInstantiation,

ExplicitTemplateArgumentSubstitution,

DeducedTemplateArgumentSubstitution,

PriorTemplateArgumentSubstitution,

DefaultTemplateArgumentChecking,

ExceptionSpecInstantiation,

DeclaringSpecialMember,

DefiningSynthesizedFunction,

Memoization
} Kind;

bool SavedInNonInstantiationSFINAEContext;

SourceLocation PointOfInstantiation;

Decl *Entity;

NamedDecl *Template;

const TemplateArgument *TemplateArgs;

union {
unsigned NumTemplateArgs;

CXXSpecialMember SpecialMember;
};

ArrayRef<TemplateArgument> template_arguments() const {
assert(Kind != DeclaringSpecialMember);
return {TemplateArgs, NumTemplateArgs};
}

sema::TemplateDeductionInfo *DeductionInfo;

SourceRange InstantiationRange;

CodeSynthesisContext()
: Kind(TemplateInstantiation), Entity(nullptr), Template(nullptr),
TemplateArgs(nullptr), NumTemplateArgs(0), DeductionInfo(nullptr) {}

bool isInstantiationRecord() const;
};

SmallVector<CodeSynthesisContext, 16> CodeSynthesisContexts;

llvm::DenseSet<std::pair<Decl *, unsigned>> InstantiatingSpecializations;

llvm::DenseSet<QualType> InstantiatedNonDependentTypes;

SmallVector<Module*, 16> CodeSynthesisContextLookupModules;

llvm::DenseSet<Module*> LookupModulesCache;

llvm::DenseSet<Module*> &getLookupModules();

llvm::DenseMap<NamedDecl*, NamedDecl*> VisibleNamespaceCache;

bool InNonInstantiationSFINAEContext;

unsigned NonInstantiationEntries;

unsigned LastEmittedCodeSynthesisContextDepth = 0;

std::vector<std::unique_ptr<TemplateInstantiationCallback>>
TemplateInstCallbacks;

int ArgumentPackSubstitutionIndex;

class ArgumentPackSubstitutionIndexRAII {
Sema &Self;
int OldSubstitutionIndex;

public:
ArgumentPackSubstitutionIndexRAII(Sema &Self, int NewSubstitutionIndex)
: Self(Self), OldSubstitutionIndex(Self.ArgumentPackSubstitutionIndex) {
Self.ArgumentPackSubstitutionIndex = NewSubstitutionIndex;
}

~ArgumentPackSubstitutionIndexRAII() {
Self.ArgumentPackSubstitutionIndex = OldSubstitutionIndex;
}
};

friend class ArgumentPackSubstitutionRAII;

typedef llvm::DenseMap<Decl *, SmallVector<PartialDiagnosticAt, 1> >
SuppressedDiagnosticsMap;
SuppressedDiagnosticsMap SuppressedDiagnostics;

struct InstantiatingTemplate {
InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
Decl *Entity,
SourceRange InstantiationRange = SourceRange());

struct ExceptionSpecification {};
InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
FunctionDecl *Entity, ExceptionSpecification,
SourceRange InstantiationRange = SourceRange());

InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
TemplateParameter Param, TemplateDecl *Template,
ArrayRef<TemplateArgument> TemplateArgs,
SourceRange InstantiationRange = SourceRange());

InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
FunctionTemplateDecl *FunctionTemplate,
ArrayRef<TemplateArgument> TemplateArgs,
CodeSynthesisContext::SynthesisKind Kind,
sema::TemplateDeductionInfo &DeductionInfo,
SourceRange InstantiationRange = SourceRange());

InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
TemplateDecl *Template,
ArrayRef<TemplateArgument> TemplateArgs,
sema::TemplateDeductionInfo &DeductionInfo,
SourceRange InstantiationRange = SourceRange());

InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
ClassTemplatePartialSpecializationDecl *PartialSpec,
ArrayRef<TemplateArgument> TemplateArgs,
sema::TemplateDeductionInfo &DeductionInfo,
SourceRange InstantiationRange = SourceRange());

InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
VarTemplatePartialSpecializationDecl *PartialSpec,
ArrayRef<TemplateArgument> TemplateArgs,
sema::TemplateDeductionInfo &DeductionInfo,
SourceRange InstantiationRange = SourceRange());

InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
ParmVarDecl *Param,
ArrayRef<TemplateArgument> TemplateArgs,
SourceRange InstantiationRange = SourceRange());

InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
NamedDecl *Template,
NonTypeTemplateParmDecl *Param,
ArrayRef<TemplateArgument> TemplateArgs,
SourceRange InstantiationRange);

InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
NamedDecl *Template,
TemplateTemplateParmDecl *Param,
ArrayRef<TemplateArgument> TemplateArgs,
SourceRange InstantiationRange);

InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
TemplateDecl *Template,
NamedDecl *Param,
ArrayRef<TemplateArgument> TemplateArgs,
SourceRange InstantiationRange);


void Clear();

~InstantiatingTemplate() { Clear(); }

bool isInvalid() const { return Invalid; }

bool isAlreadyInstantiating() const { return AlreadyInstantiating; }

private:
Sema &SemaRef;
bool Invalid;
bool AlreadyInstantiating;
bool CheckInstantiationDepth(SourceLocation PointOfInstantiation,
SourceRange InstantiationRange);

InstantiatingTemplate(
Sema &SemaRef, CodeSynthesisContext::SynthesisKind Kind,
SourceLocation PointOfInstantiation, SourceRange InstantiationRange,
Decl *Entity, NamedDecl *Template = nullptr,
ArrayRef<TemplateArgument> TemplateArgs = None,
sema::TemplateDeductionInfo *DeductionInfo = nullptr);

InstantiatingTemplate(const InstantiatingTemplate&) = delete;

InstantiatingTemplate&
operator=(const InstantiatingTemplate&) = delete;
};

void pushCodeSynthesisContext(CodeSynthesisContext Ctx);
void popCodeSynthesisContext();

bool inTemplateInstantiation() const {
return CodeSynthesisContexts.size() > NonInstantiationEntries;
}

void PrintContextStack() {
if (!CodeSynthesisContexts.empty() &&
CodeSynthesisContexts.size() != LastEmittedCodeSynthesisContextDepth) {
PrintInstantiationStack();
LastEmittedCodeSynthesisContextDepth = CodeSynthesisContexts.size();
}
if (PragmaAttributeCurrentTargetDecl)
PrintPragmaAttributeInstantiationPoint();
}
void PrintInstantiationStack();

void PrintPragmaAttributeInstantiationPoint();

Optional<sema::TemplateDeductionInfo *> isSFINAEContext() const;

bool isUnevaluatedContext() const {
assert(!ExprEvalContexts.empty() &&
"Must be in an expression evaluation context");
return ExprEvalContexts.back().isUnevaluated();
}

class SFINAETrap {
Sema &SemaRef;
unsigned PrevSFINAEErrors;
bool PrevInNonInstantiationSFINAEContext;
bool PrevAccessCheckingSFINAE;
bool PrevLastDiagnosticIgnored;

public:
explicit SFINAETrap(Sema &SemaRef, bool AccessCheckingSFINAE = false)
: SemaRef(SemaRef), PrevSFINAEErrors(SemaRef.NumSFINAEErrors),
PrevInNonInstantiationSFINAEContext(
SemaRef.InNonInstantiationSFINAEContext),
PrevAccessCheckingSFINAE(SemaRef.AccessCheckingSFINAE),
PrevLastDiagnosticIgnored(
SemaRef.getDiagnostics().isLastDiagnosticIgnored())
{
if (!SemaRef.isSFINAEContext())
SemaRef.InNonInstantiationSFINAEContext = true;
SemaRef.AccessCheckingSFINAE = AccessCheckingSFINAE;
}

~SFINAETrap() {
SemaRef.NumSFINAEErrors = PrevSFINAEErrors;
SemaRef.InNonInstantiationSFINAEContext
= PrevInNonInstantiationSFINAEContext;
SemaRef.AccessCheckingSFINAE = PrevAccessCheckingSFINAE;
SemaRef.getDiagnostics().setLastDiagnosticIgnored(
PrevLastDiagnosticIgnored);
}

bool hasErrorOccurred() const {
return SemaRef.NumSFINAEErrors > PrevSFINAEErrors;
}
};

class TentativeAnalysisScope {
Sema &SemaRef;
SFINAETrap Trap;
bool PrevDisableTypoCorrection;
public:
explicit TentativeAnalysisScope(Sema &SemaRef)
: SemaRef(SemaRef), Trap(SemaRef, true),
PrevDisableTypoCorrection(SemaRef.DisableTypoCorrection) {
SemaRef.DisableTypoCorrection = true;
}
~TentativeAnalysisScope() {
SemaRef.DisableTypoCorrection = PrevDisableTypoCorrection;
}
};

LocalInstantiationScope *CurrentInstantiationScope;

bool DisableTypoCorrection;

unsigned TyposCorrected;

typedef llvm::SmallSet<SourceLocation, 2> SrcLocSet;
typedef llvm::DenseMap<IdentifierInfo *, SrcLocSet> IdentifierSourceLocations;

IdentifierSourceLocations TypoCorrectionFailures;

sema::AnalysisBasedWarnings AnalysisWarnings;
threadSafety::BeforeSet *ThreadSafetyDeclCache;

typedef std::pair<ValueDecl *, SourceLocation> PendingImplicitInstantiation;

std::deque<PendingImplicitInstantiation> PendingInstantiations;

SmallVector<PendingImplicitInstantiation, 1> LateParsedInstantiations;

class GlobalEagerInstantiationScope {
public:
GlobalEagerInstantiationScope(Sema &S, bool Enabled)
: S(S), Enabled(Enabled) {
if (!Enabled) return;

SavedPendingInstantiations.swap(S.PendingInstantiations);
SavedVTableUses.swap(S.VTableUses);
}

void perform() {
if (Enabled) {
S.DefineUsedVTables();
S.PerformPendingInstantiations();
}
}

~GlobalEagerInstantiationScope() {
if (!Enabled) return;

assert(S.VTableUses.empty() &&
"VTableUses should be empty before it is discarded.");
S.VTableUses.swap(SavedVTableUses);

assert(S.PendingInstantiations.empty() &&
"PendingInstantiations should be empty before it is discarded.");
S.PendingInstantiations.swap(SavedPendingInstantiations);
}

private:
Sema &S;
SmallVector<VTableUse, 16> SavedVTableUses;
std::deque<PendingImplicitInstantiation> SavedPendingInstantiations;
bool Enabled;
};

std::deque<PendingImplicitInstantiation> PendingLocalImplicitInstantiations;

class LocalEagerInstantiationScope {
public:
LocalEagerInstantiationScope(Sema &S) : S(S) {
SavedPendingLocalImplicitInstantiations.swap(
S.PendingLocalImplicitInstantiations);
}

void perform() { S.PerformPendingInstantiations(true); }

~LocalEagerInstantiationScope() {
assert(S.PendingLocalImplicitInstantiations.empty() &&
"there shouldn't be any pending local implicit instantiations");
SavedPendingLocalImplicitInstantiations.swap(
S.PendingLocalImplicitInstantiations);
}

private:
Sema &S;
std::deque<PendingImplicitInstantiation>
SavedPendingLocalImplicitInstantiations;
};

class ExtParameterInfoBuilder {
SmallVector<FunctionProtoType::ExtParameterInfo, 16> Infos;
bool HasInteresting = false;

public:
void set(unsigned index, FunctionProtoType::ExtParameterInfo info) {
assert(Infos.size() <= index);
Infos.resize(index);
Infos.push_back(info);

if (!HasInteresting)
HasInteresting = (info != FunctionProtoType::ExtParameterInfo());
}

const FunctionProtoType::ExtParameterInfo *
getPointerOrNull(unsigned numParams) {
if (!HasInteresting) return nullptr;
Infos.resize(numParams);
return Infos.data();
}
};

void PerformPendingInstantiations(bool LocalOnly = false);

TypeSourceInfo *SubstType(TypeSourceInfo *T,
const MultiLevelTemplateArgumentList &TemplateArgs,
SourceLocation Loc, DeclarationName Entity,
bool AllowDeducedTST = false);

QualType SubstType(QualType T,
const MultiLevelTemplateArgumentList &TemplateArgs,
SourceLocation Loc, DeclarationName Entity);

TypeSourceInfo *SubstType(TypeLoc TL,
const MultiLevelTemplateArgumentList &TemplateArgs,
SourceLocation Loc, DeclarationName Entity);

TypeSourceInfo *SubstFunctionDeclType(TypeSourceInfo *T,
const MultiLevelTemplateArgumentList &TemplateArgs,
SourceLocation Loc,
DeclarationName Entity,
CXXRecordDecl *ThisContext,
unsigned ThisTypeQuals);
void SubstExceptionSpec(FunctionDecl *New, const FunctionProtoType *Proto,
const MultiLevelTemplateArgumentList &Args);
bool SubstExceptionSpec(SourceLocation Loc,
FunctionProtoType::ExceptionSpecInfo &ESI,
SmallVectorImpl<QualType> &ExceptionStorage,
const MultiLevelTemplateArgumentList &Args);
ParmVarDecl *SubstParmVarDecl(ParmVarDecl *D,
const MultiLevelTemplateArgumentList &TemplateArgs,
int indexAdjustment,
Optional<unsigned> NumExpansions,
bool ExpectParameterPack);
bool SubstParmTypes(SourceLocation Loc, ArrayRef<ParmVarDecl *> Params,
const FunctionProtoType::ExtParameterInfo *ExtParamInfos,
const MultiLevelTemplateArgumentList &TemplateArgs,
SmallVectorImpl<QualType> &ParamTypes,
SmallVectorImpl<ParmVarDecl *> *OutParams,
ExtParameterInfoBuilder &ParamInfos);
ExprResult SubstExpr(Expr *E,
const MultiLevelTemplateArgumentList &TemplateArgs);

bool SubstExprs(ArrayRef<Expr *> Exprs, bool IsCall,
const MultiLevelTemplateArgumentList &TemplateArgs,
SmallVectorImpl<Expr *> &Outputs);

StmtResult SubstStmt(Stmt *S,
const MultiLevelTemplateArgumentList &TemplateArgs);

TemplateParameterList *
SubstTemplateParams(TemplateParameterList *Params, DeclContext *Owner,
const MultiLevelTemplateArgumentList &TemplateArgs);

Decl *SubstDecl(Decl *D, DeclContext *Owner,
const MultiLevelTemplateArgumentList &TemplateArgs);

ExprResult SubstInitializer(Expr *E,
const MultiLevelTemplateArgumentList &TemplateArgs,
bool CXXDirectInit);

bool
SubstBaseSpecifiers(CXXRecordDecl *Instantiation,
CXXRecordDecl *Pattern,
const MultiLevelTemplateArgumentList &TemplateArgs);

bool
InstantiateClass(SourceLocation PointOfInstantiation,
CXXRecordDecl *Instantiation, CXXRecordDecl *Pattern,
const MultiLevelTemplateArgumentList &TemplateArgs,
TemplateSpecializationKind TSK,
bool Complain = true);

bool InstantiateEnum(SourceLocation PointOfInstantiation,
EnumDecl *Instantiation, EnumDecl *Pattern,
const MultiLevelTemplateArgumentList &TemplateArgs,
TemplateSpecializationKind TSK);

bool InstantiateInClassInitializer(
SourceLocation PointOfInstantiation, FieldDecl *Instantiation,
FieldDecl *Pattern, const MultiLevelTemplateArgumentList &TemplateArgs);

struct LateInstantiatedAttribute {
const Attr *TmplAttr;
LocalInstantiationScope *Scope;
Decl *NewDecl;

LateInstantiatedAttribute(const Attr *A, LocalInstantiationScope *S,
Decl *D)
: TmplAttr(A), Scope(S), NewDecl(D)
{ }
};
typedef SmallVector<LateInstantiatedAttribute, 16> LateInstantiatedAttrVec;

void InstantiateAttrs(const MultiLevelTemplateArgumentList &TemplateArgs,
const Decl *Pattern, Decl *Inst,
LateInstantiatedAttrVec *LateAttrs = nullptr,
LocalInstantiationScope *OuterMostScope = nullptr);

void
InstantiateAttrsForDecl(const MultiLevelTemplateArgumentList &TemplateArgs,
const Decl *Pattern, Decl *Inst,
LateInstantiatedAttrVec *LateAttrs = nullptr,
LocalInstantiationScope *OuterMostScope = nullptr);

bool usesPartialOrExplicitSpecialization(
SourceLocation Loc, ClassTemplateSpecializationDecl *ClassTemplateSpec);

bool
InstantiateClassTemplateSpecialization(SourceLocation PointOfInstantiation,
ClassTemplateSpecializationDecl *ClassTemplateSpec,
TemplateSpecializationKind TSK,
bool Complain = true);

void InstantiateClassMembers(SourceLocation PointOfInstantiation,
CXXRecordDecl *Instantiation,
const MultiLevelTemplateArgumentList &TemplateArgs,
TemplateSpecializationKind TSK);

void InstantiateClassTemplateSpecializationMembers(
SourceLocation PointOfInstantiation,
ClassTemplateSpecializationDecl *ClassTemplateSpec,
TemplateSpecializationKind TSK);

NestedNameSpecifierLoc
SubstNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS,
const MultiLevelTemplateArgumentList &TemplateArgs);

DeclarationNameInfo
SubstDeclarationNameInfo(const DeclarationNameInfo &NameInfo,
const MultiLevelTemplateArgumentList &TemplateArgs);
TemplateName
SubstTemplateName(NestedNameSpecifierLoc QualifierLoc, TemplateName Name,
SourceLocation Loc,
const MultiLevelTemplateArgumentList &TemplateArgs);
bool Subst(const TemplateArgumentLoc *Args, unsigned NumArgs,
TemplateArgumentListInfo &Result,
const MultiLevelTemplateArgumentList &TemplateArgs);

void InstantiateExceptionSpec(SourceLocation PointOfInstantiation,
FunctionDecl *Function);
FunctionDecl *InstantiateFunctionDeclaration(FunctionTemplateDecl *FTD,
const TemplateArgumentList *Args,
SourceLocation Loc);
void InstantiateFunctionDefinition(SourceLocation PointOfInstantiation,
FunctionDecl *Function,
bool Recursive = false,
bool DefinitionRequired = false,
bool AtEndOfTU = false);
VarTemplateSpecializationDecl *BuildVarTemplateInstantiation(
VarTemplateDecl *VarTemplate, VarDecl *FromVar,
const TemplateArgumentList &TemplateArgList,
const TemplateArgumentListInfo &TemplateArgsInfo,
SmallVectorImpl<TemplateArgument> &Converted,
SourceLocation PointOfInstantiation, void *InsertPos,
LateInstantiatedAttrVec *LateAttrs = nullptr,
LocalInstantiationScope *StartingScope = nullptr);
VarTemplateSpecializationDecl *CompleteVarTemplateSpecializationDecl(
VarTemplateSpecializationDecl *VarSpec, VarDecl *PatternDecl,
const MultiLevelTemplateArgumentList &TemplateArgs);
void
BuildVariableInstantiation(VarDecl *NewVar, VarDecl *OldVar,
const MultiLevelTemplateArgumentList &TemplateArgs,
LateInstantiatedAttrVec *LateAttrs,
DeclContext *Owner,
LocalInstantiationScope *StartingScope,
bool InstantiatingVarTemplate = false);
void InstantiateVariableInitializer(
VarDecl *Var, VarDecl *OldVar,
const MultiLevelTemplateArgumentList &TemplateArgs);
void InstantiateVariableDefinition(SourceLocation PointOfInstantiation,
VarDecl *Var, bool Recursive = false,
bool DefinitionRequired = false,
bool AtEndOfTU = false);

void InstantiateMemInitializers(CXXConstructorDecl *New,
const CXXConstructorDecl *Tmpl,
const MultiLevelTemplateArgumentList &TemplateArgs);

NamedDecl *FindInstantiatedDecl(SourceLocation Loc, NamedDecl *D,
const MultiLevelTemplateArgumentList &TemplateArgs,
bool FindingInstantiatedContext = false);
DeclContext *FindInstantiatedContext(SourceLocation Loc, DeclContext *DC,
const MultiLevelTemplateArgumentList &TemplateArgs);

enum ObjCContainerKind {
OCK_None = -1,
OCK_Interface = 0,
OCK_Protocol,
OCK_Category,
OCK_ClassExtension,
OCK_Implementation,
OCK_CategoryImplementation
};
ObjCContainerKind getObjCContainerKind() const;

DeclResult actOnObjCTypeParam(Scope *S,
ObjCTypeParamVariance variance,
SourceLocation varianceLoc,
unsigned index,
IdentifierInfo *paramName,
SourceLocation paramLoc,
SourceLocation colonLoc,
ParsedType typeBound);

ObjCTypeParamList *actOnObjCTypeParamList(Scope *S, SourceLocation lAngleLoc,
ArrayRef<Decl *> typeParams,
SourceLocation rAngleLoc);
void popObjCTypeParamList(Scope *S, ObjCTypeParamList *typeParamList);

Decl *ActOnStartClassInterface(
Scope *S, SourceLocation AtInterfaceLoc, IdentifierInfo *ClassName,
SourceLocation ClassLoc, ObjCTypeParamList *typeParamList,
IdentifierInfo *SuperName, SourceLocation SuperLoc,
ArrayRef<ParsedType> SuperTypeArgs, SourceRange SuperTypeArgsRange,
Decl *const *ProtoRefs, unsigned NumProtoRefs,
const SourceLocation *ProtoLocs, SourceLocation EndProtoLoc,
const ParsedAttributesView &AttrList);

void ActOnSuperClassOfClassInterface(Scope *S,
SourceLocation AtInterfaceLoc,
ObjCInterfaceDecl *IDecl,
IdentifierInfo *ClassName,
SourceLocation ClassLoc,
IdentifierInfo *SuperName,
SourceLocation SuperLoc,
ArrayRef<ParsedType> SuperTypeArgs,
SourceRange SuperTypeArgsRange);

void ActOnTypedefedProtocols(SmallVectorImpl<Decl *> &ProtocolRefs,
SmallVectorImpl<SourceLocation> &ProtocolLocs,
IdentifierInfo *SuperName,
SourceLocation SuperLoc);

Decl *ActOnCompatibilityAlias(
SourceLocation AtCompatibilityAliasLoc,
IdentifierInfo *AliasName,  SourceLocation AliasLocation,
IdentifierInfo *ClassName, SourceLocation ClassLocation);

bool CheckForwardProtocolDeclarationForCircularDependency(
IdentifierInfo *PName,
SourceLocation &PLoc, SourceLocation PrevLoc,
const ObjCList<ObjCProtocolDecl> &PList);

Decl *ActOnStartProtocolInterface(
SourceLocation AtProtoInterfaceLoc, IdentifierInfo *ProtocolName,
SourceLocation ProtocolLoc, Decl *const *ProtoRefNames,
unsigned NumProtoRefs, const SourceLocation *ProtoLocs,
SourceLocation EndProtoLoc, const ParsedAttributesView &AttrList);

Decl *ActOnStartCategoryInterface(
SourceLocation AtInterfaceLoc, IdentifierInfo *ClassName,
SourceLocation ClassLoc, ObjCTypeParamList *typeParamList,
IdentifierInfo *CategoryName, SourceLocation CategoryLoc,
Decl *const *ProtoRefs, unsigned NumProtoRefs,
const SourceLocation *ProtoLocs, SourceLocation EndProtoLoc,
const ParsedAttributesView &AttrList);

Decl *ActOnStartClassImplementation(
SourceLocation AtClassImplLoc,
IdentifierInfo *ClassName, SourceLocation ClassLoc,
IdentifierInfo *SuperClassname,
SourceLocation SuperClassLoc);

Decl *ActOnStartCategoryImplementation(SourceLocation AtCatImplLoc,
IdentifierInfo *ClassName,
SourceLocation ClassLoc,
IdentifierInfo *CatName,
SourceLocation CatLoc);

DeclGroupPtrTy ActOnFinishObjCImplementation(Decl *ObjCImpDecl,
ArrayRef<Decl *> Decls);

DeclGroupPtrTy ActOnForwardClassDeclaration(SourceLocation Loc,
IdentifierInfo **IdentList,
SourceLocation *IdentLocs,
ArrayRef<ObjCTypeParamList *> TypeParamLists,
unsigned NumElts);

DeclGroupPtrTy
ActOnForwardProtocolDeclaration(SourceLocation AtProtoclLoc,
ArrayRef<IdentifierLocPair> IdentList,
const ParsedAttributesView &attrList);

void FindProtocolDeclaration(bool WarnOnDeclarations, bool ForObjCContainer,
ArrayRef<IdentifierLocPair> ProtocolId,
SmallVectorImpl<Decl *> &Protocols);

void DiagnoseTypeArgsAndProtocols(IdentifierInfo *ProtocolId,
SourceLocation ProtocolLoc,
IdentifierInfo *TypeArgId,
SourceLocation TypeArgLoc,
bool SelectProtocolFirst = false);

void actOnObjCTypeArgsOrProtocolQualifiers(
Scope *S,
ParsedType baseType,
SourceLocation lAngleLoc,
ArrayRef<IdentifierInfo *> identifiers,
ArrayRef<SourceLocation> identifierLocs,
SourceLocation rAngleLoc,
SourceLocation &typeArgsLAngleLoc,
SmallVectorImpl<ParsedType> &typeArgs,
SourceLocation &typeArgsRAngleLoc,
SourceLocation &protocolLAngleLoc,
SmallVectorImpl<Decl *> &protocols,
SourceLocation &protocolRAngleLoc,
bool warnOnIncompleteProtocols);

TypeResult actOnObjCProtocolQualifierType(
SourceLocation lAngleLoc,
ArrayRef<Decl *> protocols,
ArrayRef<SourceLocation> protocolLocs,
SourceLocation rAngleLoc);

TypeResult actOnObjCTypeArgsAndProtocolQualifiers(
Scope *S,
SourceLocation Loc,
ParsedType BaseType,
SourceLocation TypeArgsLAngleLoc,
ArrayRef<ParsedType> TypeArgs,
SourceLocation TypeArgsRAngleLoc,
SourceLocation ProtocolLAngleLoc,
ArrayRef<Decl *> Protocols,
ArrayRef<SourceLocation> ProtocolLocs,
SourceLocation ProtocolRAngleLoc);

QualType BuildObjCTypeParamType(const ObjCTypeParamDecl *Decl,
SourceLocation ProtocolLAngleLoc,
ArrayRef<ObjCProtocolDecl *> Protocols,
ArrayRef<SourceLocation> ProtocolLocs,
SourceLocation ProtocolRAngleLoc,
bool FailOnError = false);

QualType BuildObjCObjectType(QualType BaseType,
SourceLocation Loc,
SourceLocation TypeArgsLAngleLoc,
ArrayRef<TypeSourceInfo *> TypeArgs,
SourceLocation TypeArgsRAngleLoc,
SourceLocation ProtocolLAngleLoc,
ArrayRef<ObjCProtocolDecl *> Protocols,
ArrayRef<SourceLocation> ProtocolLocs,
SourceLocation ProtocolRAngleLoc,
bool FailOnError = false);

bool checkObjCKindOfType(QualType &type, SourceLocation loc);

void CheckObjCPropertyAttributes(Decl *PropertyPtrTy,
SourceLocation Loc,
unsigned &Attributes,
bool propertyInPrimaryClass);

void ProcessPropertyDecl(ObjCPropertyDecl *property);


void DiagnosePropertyMismatch(ObjCPropertyDecl *Property,
ObjCPropertyDecl *SuperProperty,
const IdentifierInfo *Name,
bool OverridingProtocolProperty);

void DiagnoseClassExtensionDupMethods(ObjCCategoryDecl *CAT,
ObjCInterfaceDecl *ID);

Decl *ActOnAtEnd(Scope *S, SourceRange AtEnd,
ArrayRef<Decl *> allMethods = None,
ArrayRef<DeclGroupPtrTy> allTUVars = None);

Decl *ActOnProperty(Scope *S, SourceLocation AtLoc,
SourceLocation LParenLoc,
FieldDeclarator &FD, ObjCDeclSpec &ODS,
Selector GetterSel, Selector SetterSel,
tok::ObjCKeywordKind MethodImplKind,
DeclContext *lexicalDC = nullptr);

Decl *ActOnPropertyImplDecl(Scope *S,
SourceLocation AtLoc,
SourceLocation PropertyLoc,
bool ImplKind,
IdentifierInfo *PropertyId,
IdentifierInfo *PropertyIvar,
SourceLocation PropertyIvarLoc,
ObjCPropertyQueryKind QueryKind);

enum ObjCSpecialMethodKind {
OSMK_None,
OSMK_Alloc,
OSMK_New,
OSMK_Copy,
OSMK_RetainingInit,
OSMK_NonRetainingInit
};

struct ObjCArgInfo {
IdentifierInfo *Name;
SourceLocation NameLoc;
ParsedType Type;
ObjCDeclSpec DeclSpec;

ParsedAttributesView ArgAttrs;
};

Decl *ActOnMethodDeclaration(
Scope *S,
SourceLocation BeginLoc, 
SourceLocation EndLoc,   
tok::TokenKind MethodType, ObjCDeclSpec &ReturnQT, ParsedType ReturnType,
ArrayRef<SourceLocation> SelectorLocs, Selector Sel,
ObjCArgInfo *ArgInfo, DeclaratorChunk::ParamInfo *CParamInfo,
unsigned CNumArgs, 
const ParsedAttributesView &AttrList, tok::ObjCKeywordKind MethodImplKind,
bool isVariadic, bool MethodDefinition);

ObjCMethodDecl *LookupMethodInQualifiedType(Selector Sel,
const ObjCObjectPointerType *OPT,
bool IsInstance);
ObjCMethodDecl *LookupMethodInObjectType(Selector Sel, QualType Ty,
bool IsInstance);

bool CheckARCMethodDecl(ObjCMethodDecl *method);
bool inferObjCARCLifetime(ValueDecl *decl);

ExprResult
HandleExprPropertyRefExpr(const ObjCObjectPointerType *OPT,
Expr *BaseExpr,
SourceLocation OpLoc,
DeclarationName MemberName,
SourceLocation MemberLoc,
SourceLocation SuperLoc, QualType SuperType,
bool Super);

ExprResult
ActOnClassPropertyRefExpr(IdentifierInfo &receiverName,
IdentifierInfo &propertyName,
SourceLocation receiverNameLoc,
SourceLocation propertyNameLoc);

ObjCMethodDecl *tryCaptureObjCSelf(SourceLocation Loc);

enum ObjCMessageKind {
ObjCSuperMessage,
ObjCInstanceMessage,
ObjCClassMessage
};

ObjCMessageKind getObjCMessageKind(Scope *S,
IdentifierInfo *Name,
SourceLocation NameLoc,
bool IsSuper,
bool HasTrailingDot,
ParsedType &ReceiverType);

ExprResult ActOnSuperMessage(Scope *S, SourceLocation SuperLoc,
Selector Sel,
SourceLocation LBracLoc,
ArrayRef<SourceLocation> SelectorLocs,
SourceLocation RBracLoc,
MultiExprArg Args);

ExprResult BuildClassMessage(TypeSourceInfo *ReceiverTypeInfo,
QualType ReceiverType,
SourceLocation SuperLoc,
Selector Sel,
ObjCMethodDecl *Method,
SourceLocation LBracLoc,
ArrayRef<SourceLocation> SelectorLocs,
SourceLocation RBracLoc,
MultiExprArg Args,
bool isImplicit = false);

ExprResult BuildClassMessageImplicit(QualType ReceiverType,
bool isSuperReceiver,
SourceLocation Loc,
Selector Sel,
ObjCMethodDecl *Method,
MultiExprArg Args);

ExprResult ActOnClassMessage(Scope *S,
ParsedType Receiver,
Selector Sel,
SourceLocation LBracLoc,
ArrayRef<SourceLocation> SelectorLocs,
SourceLocation RBracLoc,
MultiExprArg Args);

ExprResult BuildInstanceMessage(Expr *Receiver,
QualType ReceiverType,
SourceLocation SuperLoc,
Selector Sel,
ObjCMethodDecl *Method,
SourceLocation LBracLoc,
ArrayRef<SourceLocation> SelectorLocs,
SourceLocation RBracLoc,
MultiExprArg Args,
bool isImplicit = false);

ExprResult BuildInstanceMessageImplicit(Expr *Receiver,
QualType ReceiverType,
SourceLocation Loc,
Selector Sel,
ObjCMethodDecl *Method,
MultiExprArg Args);

ExprResult ActOnInstanceMessage(Scope *S,
Expr *Receiver,
Selector Sel,
SourceLocation LBracLoc,
ArrayRef<SourceLocation> SelectorLocs,
SourceLocation RBracLoc,
MultiExprArg Args);

ExprResult BuildObjCBridgedCast(SourceLocation LParenLoc,
ObjCBridgeCastKind Kind,
SourceLocation BridgeKeywordLoc,
TypeSourceInfo *TSInfo,
Expr *SubExpr);

ExprResult ActOnObjCBridgedCast(Scope *S,
SourceLocation LParenLoc,
ObjCBridgeCastKind Kind,
SourceLocation BridgeKeywordLoc,
ParsedType Type,
SourceLocation RParenLoc,
Expr *SubExpr);

void CheckTollFreeBridgeCast(QualType castType, Expr *castExpr);

void CheckObjCBridgeRelatedCast(QualType castType, Expr *castExpr);

bool CheckTollFreeBridgeStaticCast(QualType castType, Expr *castExpr,
CastKind &Kind);

bool checkObjCBridgeRelatedComponents(SourceLocation Loc,
QualType DestType, QualType SrcType,
ObjCInterfaceDecl *&RelatedClass,
ObjCMethodDecl *&ClassMethod,
ObjCMethodDecl *&InstanceMethod,
TypedefNameDecl *&TDNDecl,
bool CfToNs, bool Diagnose = true);

bool CheckObjCBridgeRelatedConversions(SourceLocation Loc,
QualType DestType, QualType SrcType,
Expr *&SrcExpr, bool Diagnose = true);

bool ConversionToObjCStringLiteralCheck(QualType DstType, Expr *&SrcExpr,
bool Diagnose = true);

bool checkInitMethod(ObjCMethodDecl *method, QualType receiverTypeIfCall);

void CheckObjCMethodOverride(ObjCMethodDecl *NewMethod,
const ObjCMethodDecl *Overridden);

enum ResultTypeCompatibilityKind {
RTC_Compatible,
RTC_Incompatible,
RTC_Unknown
};

void CheckObjCMethodOverrides(ObjCMethodDecl *ObjCMethod,
ObjCInterfaceDecl *CurrentClass,
ResultTypeCompatibilityKind RTC);

enum PragmaOptionsAlignKind {
POAK_Native,  
POAK_Natural, 
POAK_Packed,  
POAK_Power,   
POAK_Mac68k,  
POAK_Reset    
};

void ActOnPragmaClangSection(SourceLocation PragmaLoc,
PragmaClangSectionAction Action,
PragmaClangSectionKind SecKind, StringRef SecName);

void ActOnPragmaOptionsAlign(PragmaOptionsAlignKind Kind,
SourceLocation PragmaLoc);

void ActOnPragmaPack(SourceLocation PragmaLoc, PragmaMsStackAction Action,
StringRef SlotLabel, Expr *Alignment);

enum class PragmaPackDiagnoseKind {
NonDefaultStateAtInclude,
ChangedStateAtExit
};

void DiagnoseNonDefaultPragmaPack(PragmaPackDiagnoseKind Kind,
SourceLocation IncludeLoc);
void DiagnoseUnterminatedPragmaPack();

void ActOnPragmaMSStruct(PragmaMSStructKind Kind);

void ActOnPragmaMSComment(SourceLocation CommentLoc, PragmaMSCommentKind Kind,
StringRef Arg);

void ActOnPragmaMSPointersToMembers(
LangOptions::PragmaMSPointersToMembersKind Kind,
SourceLocation PragmaLoc);

void ActOnPragmaMSVtorDisp(PragmaMsStackAction Action,
SourceLocation PragmaLoc,
MSVtorDispAttr::Mode Value);

enum PragmaSectionKind {
PSK_DataSeg,
PSK_BSSSeg,
PSK_ConstSeg,
PSK_CodeSeg,
};

bool UnifySection(StringRef SectionName,
int SectionFlags,
DeclaratorDecl *TheDecl);
bool UnifySection(StringRef SectionName,
int SectionFlags,
SourceLocation PragmaSectionLocation);

void ActOnPragmaMSSeg(SourceLocation PragmaLocation,
PragmaMsStackAction Action,
llvm::StringRef StackSlotLabel,
StringLiteral *SegmentName,
llvm::StringRef PragmaName);

void ActOnPragmaMSSection(SourceLocation PragmaLocation,
int SectionFlags, StringLiteral *SegmentName);

void ActOnPragmaMSInitSeg(SourceLocation PragmaLocation,
StringLiteral *SegmentName);

void ActOnPragmaDump(Scope *S, SourceLocation Loc, IdentifierInfo *II);

void ActOnPragmaDetectMismatch(SourceLocation Loc, StringRef Name,
StringRef Value);

void ActOnPragmaUnused(const Token &Identifier,
Scope *curScope,
SourceLocation PragmaLoc);

void ActOnPragmaVisibility(const IdentifierInfo* VisType,
SourceLocation PragmaLoc);

NamedDecl *DeclClonePragmaWeak(NamedDecl *ND, IdentifierInfo *II,
SourceLocation Loc);
void DeclApplyPragmaWeak(Scope *S, NamedDecl *ND, WeakInfo &W);

void ActOnPragmaWeakID(IdentifierInfo* WeakName,
SourceLocation PragmaLoc,
SourceLocation WeakNameLoc);

void ActOnPragmaRedefineExtname(IdentifierInfo* WeakName,
IdentifierInfo* AliasName,
SourceLocation PragmaLoc,
SourceLocation WeakNameLoc,
SourceLocation AliasNameLoc);

void ActOnPragmaWeakAlias(IdentifierInfo* WeakName,
IdentifierInfo* AliasName,
SourceLocation PragmaLoc,
SourceLocation WeakNameLoc,
SourceLocation AliasNameLoc);

void ActOnPragmaFPContract(LangOptions::FPContractModeKind FPC);

void AddAlignmentAttributesForRecord(RecordDecl *RD);

void AddMsStructLayoutForRecord(RecordDecl *RD);

void FreePackedContext();

void PushNamespaceVisibilityAttr(const VisibilityAttr *Attr,
SourceLocation Loc);

void AddPushedVisibilityAttribute(Decl *RD);

void PopPragmaVisibility(bool IsNamespaceEnd, SourceLocation EndLoc);

void FreeVisContext();

void AddCFAuditedAttribute(Decl *D);

void ActOnPragmaAttributePush(ParsedAttr &Attribute, SourceLocation PragmaLoc,
attr::ParsedSubjectMatchRuleSet Rules);

void ActOnPragmaAttributePop(SourceLocation PragmaLoc);

void AddPragmaAttributes(Scope *S, Decl *D);

void DiagnoseUnterminatedPragmaAttribute();

void ActOnPragmaOptimize(bool On, SourceLocation PragmaLoc);

SourceLocation getOptimizeOffPragmaLocation() const {
return OptimizeOffPragmaLocation;
}

void AddRangeBasedOptnone(FunctionDecl *FD);

void AddOptnoneAttributeIfNoConflicts(FunctionDecl *FD, SourceLocation Loc);

void AddAlignedAttr(SourceRange AttrRange, Decl *D, Expr *E,
unsigned SpellingListIndex, bool IsPackExpansion);
void AddAlignedAttr(SourceRange AttrRange, Decl *D, TypeSourceInfo *T,
unsigned SpellingListIndex, bool IsPackExpansion);

void AddAssumeAlignedAttr(SourceRange AttrRange, Decl *D, Expr *E, Expr *OE,
unsigned SpellingListIndex);

void AddAllocAlignAttr(SourceRange AttrRange, Decl *D, Expr *ParamExpr,
unsigned SpellingListIndex);

void AddAlignValueAttr(SourceRange AttrRange, Decl *D, Expr *E,
unsigned SpellingListIndex);

void AddLaunchBoundsAttr(SourceRange AttrRange, Decl *D, Expr *MaxThreads,
Expr *MinBlocks, unsigned SpellingListIndex);

void AddModeAttr(SourceRange AttrRange, Decl *D, IdentifierInfo *Name,
unsigned SpellingListIndex, bool InInstantiation = false);

void AddParameterABIAttr(SourceRange AttrRange, Decl *D,
ParameterABI ABI, unsigned SpellingListIndex);

void AddNSConsumedAttr(SourceRange AttrRange, Decl *D,
unsigned SpellingListIndex, bool isNSConsumed,
bool isTemplateInstantiation);

bool checkNSReturnsRetainedReturnType(SourceLocation loc, QualType type);

bool ActOnCoroutineBodyStart(Scope *S, SourceLocation KwLoc,
StringRef Keyword);
ExprResult ActOnCoawaitExpr(Scope *S, SourceLocation KwLoc, Expr *E);
ExprResult ActOnCoyieldExpr(Scope *S, SourceLocation KwLoc, Expr *E);
StmtResult ActOnCoreturnStmt(Scope *S, SourceLocation KwLoc, Expr *E);

ExprResult BuildResolvedCoawaitExpr(SourceLocation KwLoc, Expr *E,
bool IsImplicit = false);
ExprResult BuildUnresolvedCoawaitExpr(SourceLocation KwLoc, Expr *E,
UnresolvedLookupExpr* Lookup);
ExprResult BuildCoyieldExpr(SourceLocation KwLoc, Expr *E);
StmtResult BuildCoreturnStmt(SourceLocation KwLoc, Expr *E,
bool IsImplicit = false);
StmtResult BuildCoroutineBodyStmt(CoroutineBodyStmt::CtorArgs);
bool buildCoroutineParameterMoves(SourceLocation Loc);
VarDecl *buildCoroutinePromise(SourceLocation Loc);
void CheckCompletedCoroutineBody(FunctionDecl *FD, Stmt *&Body);
ClassTemplateDecl *lookupCoroutineTraits(SourceLocation KwLoc,
SourceLocation FuncLoc);

private:
std::string CurrOpenCLExtension;
llvm::DenseMap<const Type*, std::set<std::string>> OpenCLTypeExtMap;
llvm::DenseMap<const Decl*, std::set<std::string>> OpenCLDeclExtMap;
public:
llvm::StringRef getCurrentOpenCLExtension() const {
return CurrOpenCLExtension;
}
void setCurrentOpenCLExtension(llvm::StringRef Ext) {
CurrOpenCLExtension = Ext;
}

void setOpenCLExtensionForType(QualType T, llvm::StringRef Exts);

void setOpenCLExtensionForDecl(Decl *FD, llvm::StringRef Exts);

void setCurrentOpenCLExtensionForType(QualType T);

void setCurrentOpenCLExtensionForDecl(Decl *FD);

bool isOpenCLDisabledDecl(Decl *FD);

bool checkOpenCLDisabledTypeDeclSpec(const DeclSpec &DS, QualType T);

bool checkOpenCLDisabledDecl(const NamedDecl &D, const Expr &E);

private:
void *VarDataSharingAttributesStack;
bool IsInOpenMPDeclareTargetContext = false;
void InitDataSharingAttributesStack();
void DestroyDataSharingAttributesStack();
ExprResult
VerifyPositiveIntegerConstantInClause(Expr *Op, OpenMPClauseKind CKind,
bool StrictlyPositive = true);
unsigned getOpenMPNestingLevel() const;

void adjustOpenMPTargetScopeIndex(unsigned &FunctionScopesIndex,
unsigned Level) const;

void pushOpenMPFunctionRegion();

void popOpenMPFunctionRegion(const sema::FunctionScopeInfo *OldFSI);

template <typename T, typename DiagLocT, typename DiagInfoT, typename MapT>
bool checkOpenCLDisabledTypeOrDecl(T D, DiagLocT DiagLoc, DiagInfoT DiagInfo,
MapT &Map, unsigned Selector = 0,
SourceRange SrcRange = SourceRange());

public:
bool isOpenMPCapturedByRef(const ValueDecl *D, unsigned Level) const;

VarDecl *isOpenMPCapturedDecl(ValueDecl *D) const;
ExprResult getOpenMPCapturedExpr(VarDecl *Capture, ExprValueKind VK,
ExprObjectKind OK, SourceLocation Loc);

bool isOpenMPPrivateDecl(const ValueDecl *D, unsigned Level) const;

void setOpenMPCaptureKind(FieldDecl *FD, const ValueDecl *D, unsigned Level);

bool isOpenMPTargetCapturedDecl(const ValueDecl *D, unsigned Level) const;

ExprResult PerformOpenMPImplicitIntegerConversion(SourceLocation OpLoc,
Expr *Op);
void StartOpenMPDSABlock(OpenMPDirectiveKind K,
const DeclarationNameInfo &DirName, Scope *CurScope,
SourceLocation Loc);
void StartOpenMPClause(OpenMPClauseKind K);
void EndOpenMPClause();
void EndOpenMPDSABlock(Stmt *CurDirective);

void ActOnOpenMPLoopInitialization(SourceLocation ForLoc, Stmt *Init);

ExprResult ActOnOpenMPIdExpression(Scope *CurScope,
CXXScopeSpec &ScopeSpec,
const DeclarationNameInfo &Id);
DeclGroupPtrTy ActOnOpenMPThreadprivateDirective(
SourceLocation Loc,
ArrayRef<Expr *> VarList);
OMPThreadPrivateDecl *CheckOMPThreadPrivateDecl(SourceLocation Loc,
ArrayRef<Expr *> VarList);
QualType ActOnOpenMPDeclareReductionType(SourceLocation TyLoc,
TypeResult ParsedType);
DeclGroupPtrTy ActOnOpenMPDeclareReductionDirectiveStart(
Scope *S, DeclContext *DC, DeclarationName Name,
ArrayRef<std::pair<QualType, SourceLocation>> ReductionTypes,
AccessSpecifier AS, Decl *PrevDeclInScope = nullptr);
void ActOnOpenMPDeclareReductionCombinerStart(Scope *S, Decl *D);
void ActOnOpenMPDeclareReductionCombinerEnd(Decl *D, Expr *Combiner);
VarDecl *ActOnOpenMPDeclareReductionInitializerStart(Scope *S, Decl *D);
void ActOnOpenMPDeclareReductionInitializerEnd(Decl *D, Expr *Initializer,
VarDecl *OmpPrivParm);
DeclGroupPtrTy ActOnOpenMPDeclareReductionDirectiveEnd(
Scope *S, DeclGroupPtrTy DeclReductions, bool IsValid);

bool ActOnStartOpenMPDeclareTargetDirective(SourceLocation Loc);
void ActOnFinishOpenMPDeclareTargetDirective();
void ActOnOpenMPDeclareTargetName(Scope *CurScope, CXXScopeSpec &ScopeSpec,
const DeclarationNameInfo &Id,
OMPDeclareTargetDeclAttr::MapTypeTy MT,
NamedDeclSetType &SameDirectiveDecls);
void checkDeclIsAllowedInOpenMPTarget(Expr *E, Decl *D,
SourceLocation IdLoc = SourceLocation());
bool isInOpenMPDeclareTargetContext() const {
return IsInOpenMPDeclareTargetContext;
}
bool isInOpenMPTargetExecutionDirective() const;
bool shouldDiagnoseTargetSupportFromOpenMP() const {
return !getLangOpts().OpenMPIsDevice || isInOpenMPDeclareTargetContext() ||
isInOpenMPTargetExecutionDirective();
}

static int getOpenMPCaptureLevels(OpenMPDirectiveKind Kind);

void ActOnOpenMPRegionStart(OpenMPDirectiveKind DKind, Scope *CurScope);
StmtResult ActOnOpenMPRegionEnd(StmtResult S, ArrayRef<OMPClause *> Clauses);
StmtResult ActOnOpenMPExecutableDirective(
OpenMPDirectiveKind Kind, const DeclarationNameInfo &DirName,
OpenMPDirectiveKind CancelRegion, ArrayRef<OMPClause *> Clauses,
Stmt *AStmt, SourceLocation StartLoc, SourceLocation EndLoc);
StmtResult ActOnOpenMPParallelDirective(ArrayRef<OMPClause *> Clauses,
Stmt *AStmt,
SourceLocation StartLoc,
SourceLocation EndLoc);
using VarsWithInheritedDSAType =
llvm::SmallDenseMap<const ValueDecl *, const Expr *, 4>;
StmtResult
ActOnOpenMPSimdDirective(ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
SourceLocation StartLoc, SourceLocation EndLoc,
VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult
ActOnOpenMPForDirective(ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
SourceLocation StartLoc, SourceLocation EndLoc,
VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult
ActOnOpenMPForSimdDirective(ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
SourceLocation StartLoc, SourceLocation EndLoc,
VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult ActOnOpenMPSectionsDirective(ArrayRef<OMPClause *> Clauses,
Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc);
StmtResult ActOnOpenMPSectionDirective(Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc);
StmtResult ActOnOpenMPSingleDirective(ArrayRef<OMPClause *> Clauses,
Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc);
StmtResult ActOnOpenMPMasterDirective(Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc);
StmtResult ActOnOpenMPCriticalDirective(const DeclarationNameInfo &DirName,
ArrayRef<OMPClause *> Clauses,
Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc);
StmtResult ActOnOpenMPParallelForDirective(
ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult ActOnOpenMPParallelForSimdDirective(
ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult ActOnOpenMPParallelSectionsDirective(ArrayRef<OMPClause *> Clauses,
Stmt *AStmt,
SourceLocation StartLoc,
SourceLocation EndLoc);
StmtResult ActOnOpenMPTaskDirective(ArrayRef<OMPClause *> Clauses,
Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc);
StmtResult ActOnOpenMPTaskyieldDirective(SourceLocation StartLoc,
SourceLocation EndLoc);
StmtResult ActOnOpenMPBarrierDirective(SourceLocation StartLoc,
SourceLocation EndLoc);
StmtResult ActOnOpenMPTaskwaitDirective(SourceLocation StartLoc,
SourceLocation EndLoc);
StmtResult ActOnOpenMPTaskgroupDirective(ArrayRef<OMPClause *> Clauses,
Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc);
StmtResult ActOnOpenMPFlushDirective(ArrayRef<OMPClause *> Clauses,
SourceLocation StartLoc,
SourceLocation EndLoc);
StmtResult ActOnOpenMPOrderedDirective(ArrayRef<OMPClause *> Clauses,
Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc);
StmtResult ActOnOpenMPAtomicDirective(ArrayRef<OMPClause *> Clauses,
Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc);
StmtResult ActOnOpenMPTargetDirective(ArrayRef<OMPClause *> Clauses,
Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc);
StmtResult ActOnOpenMPTargetDataDirective(ArrayRef<OMPClause *> Clauses,
Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc);
StmtResult ActOnOpenMPTargetEnterDataDirective(ArrayRef<OMPClause *> Clauses,
SourceLocation StartLoc,
SourceLocation EndLoc,
Stmt *AStmt);
StmtResult ActOnOpenMPTargetExitDataDirective(ArrayRef<OMPClause *> Clauses,
SourceLocation StartLoc,
SourceLocation EndLoc,
Stmt *AStmt);
StmtResult ActOnOpenMPTargetParallelDirective(ArrayRef<OMPClause *> Clauses,
Stmt *AStmt,
SourceLocation StartLoc,
SourceLocation EndLoc);
StmtResult ActOnOpenMPTargetParallelForDirective(
ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult ActOnOpenMPTeamsDirective(ArrayRef<OMPClause *> Clauses,
Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc);
StmtResult
ActOnOpenMPCancellationPointDirective(SourceLocation StartLoc,
SourceLocation EndLoc,
OpenMPDirectiveKind CancelRegion);
StmtResult ActOnOpenMPCancelDirective(ArrayRef<OMPClause *> Clauses,
SourceLocation StartLoc,
SourceLocation EndLoc,
OpenMPDirectiveKind CancelRegion);
StmtResult
ActOnOpenMPTaskLoopDirective(ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
SourceLocation StartLoc, SourceLocation EndLoc,
VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult ActOnOpenMPTaskLoopSimdDirective(
ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult
ActOnOpenMPDistributeDirective(ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
SourceLocation StartLoc, SourceLocation EndLoc,
VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult ActOnOpenMPTargetUpdateDirective(ArrayRef<OMPClause *> Clauses,
SourceLocation StartLoc,
SourceLocation EndLoc,
Stmt *AStmt);
StmtResult ActOnOpenMPDistributeParallelForDirective(
ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult ActOnOpenMPDistributeParallelForSimdDirective(
ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult ActOnOpenMPDistributeSimdDirective(
ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult ActOnOpenMPTargetParallelForSimdDirective(
ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult
ActOnOpenMPTargetSimdDirective(ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
SourceLocation StartLoc, SourceLocation EndLoc,
VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult ActOnOpenMPTeamsDistributeDirective(
ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult ActOnOpenMPTeamsDistributeSimdDirective(
ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult ActOnOpenMPTeamsDistributeParallelForSimdDirective(
ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult ActOnOpenMPTeamsDistributeParallelForDirective(
ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult ActOnOpenMPTargetTeamsDirective(ArrayRef<OMPClause *> Clauses,
Stmt *AStmt,
SourceLocation StartLoc,
SourceLocation EndLoc);
StmtResult ActOnOpenMPTargetTeamsDistributeDirective(
ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult ActOnOpenMPTargetTeamsDistributeParallelForDirective(
ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult ActOnOpenMPTargetTeamsDistributeParallelForSimdDirective(
ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);
StmtResult ActOnOpenMPTargetTeamsDistributeSimdDirective(
ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
SourceLocation EndLoc, VarsWithInheritedDSAType &VarsWithImplicitDSA);

bool CheckOpenMPLinearModifier(OpenMPLinearClauseKind LinKind,
SourceLocation LinLoc);
bool CheckOpenMPLinearDecl(const ValueDecl *D, SourceLocation ELoc,
OpenMPLinearClauseKind LinKind, QualType Type);

DeclGroupPtrTy ActOnOpenMPDeclareSimdDirective(
DeclGroupPtrTy DG, OMPDeclareSimdDeclAttr::BranchStateTy BS,
Expr *Simdlen, ArrayRef<Expr *> Uniforms, ArrayRef<Expr *> Aligneds,
ArrayRef<Expr *> Alignments, ArrayRef<Expr *> Linears,
ArrayRef<unsigned> LinModifiers, ArrayRef<Expr *> Steps, SourceRange SR);

OMPClause *ActOnOpenMPSingleExprClause(OpenMPClauseKind Kind,
Expr *Expr,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPIfClause(OpenMPDirectiveKind NameModifier,
Expr *Condition, SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation NameModifierLoc,
SourceLocation ColonLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPFinalClause(Expr *Condition, SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPNumThreadsClause(Expr *NumThreads,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPSafelenClause(Expr *Length,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPSimdlenClause(Expr *Length, SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPCollapseClause(Expr *NumForLoops,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *
ActOnOpenMPOrderedClause(SourceLocation StartLoc, SourceLocation EndLoc,
SourceLocation LParenLoc = SourceLocation(),
Expr *NumForLoops = nullptr);
OMPClause *ActOnOpenMPGrainsizeClause(Expr *Size, SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPNumTasksClause(Expr *NumTasks, SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPHintClause(Expr *Hint, SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);

OMPClause *ActOnOpenMPSimpleClause(OpenMPClauseKind Kind,
unsigned Argument,
SourceLocation ArgumentLoc,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPDefaultClause(OpenMPDefaultClauseKind Kind,
SourceLocation KindLoc,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPProcBindClause(OpenMPProcBindClauseKind Kind,
SourceLocation KindLoc,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);

OMPClause *ActOnOpenMPSingleExprWithArgClause(
OpenMPClauseKind Kind, ArrayRef<unsigned> Arguments, Expr *Expr,
SourceLocation StartLoc, SourceLocation LParenLoc,
ArrayRef<SourceLocation> ArgumentsLoc, SourceLocation DelimLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPScheduleClause(
OpenMPScheduleClauseModifier M1, OpenMPScheduleClauseModifier M2,
OpenMPScheduleClauseKind Kind, Expr *ChunkSize, SourceLocation StartLoc,
SourceLocation LParenLoc, SourceLocation M1Loc, SourceLocation M2Loc,
SourceLocation KindLoc, SourceLocation CommaLoc, SourceLocation EndLoc);

OMPClause *ActOnOpenMPClause(OpenMPClauseKind Kind, SourceLocation StartLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPNowaitClause(SourceLocation StartLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPUntiedClause(SourceLocation StartLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPMergeableClause(SourceLocation StartLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPReadClause(SourceLocation StartLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPWriteClause(SourceLocation StartLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPUpdateClause(SourceLocation StartLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPCaptureClause(SourceLocation StartLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPSeqCstClause(SourceLocation StartLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPThreadsClause(SourceLocation StartLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPSIMDClause(SourceLocation StartLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPNogroupClause(SourceLocation StartLoc,
SourceLocation EndLoc);

OMPClause *ActOnOpenMPVarListClause(
OpenMPClauseKind Kind, ArrayRef<Expr *> Vars, Expr *TailExpr,
SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation ColonLoc, SourceLocation EndLoc,
CXXScopeSpec &ReductionIdScopeSpec,
const DeclarationNameInfo &ReductionId, OpenMPDependClauseKind DepKind,
OpenMPLinearClauseKind LinKind, OpenMPMapClauseKind MapTypeModifier,
OpenMPMapClauseKind MapType, bool IsMapTypeImplicit,
SourceLocation DepLinMapLoc);
OMPClause *ActOnOpenMPPrivateClause(ArrayRef<Expr *> VarList,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPFirstprivateClause(ArrayRef<Expr *> VarList,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPLastprivateClause(ArrayRef<Expr *> VarList,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPSharedClause(ArrayRef<Expr *> VarList,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPReductionClause(
ArrayRef<Expr *> VarList, SourceLocation StartLoc,
SourceLocation LParenLoc, SourceLocation ColonLoc, SourceLocation EndLoc,
CXXScopeSpec &ReductionIdScopeSpec,
const DeclarationNameInfo &ReductionId,
ArrayRef<Expr *> UnresolvedReductions = llvm::None);
OMPClause *ActOnOpenMPTaskReductionClause(
ArrayRef<Expr *> VarList, SourceLocation StartLoc,
SourceLocation LParenLoc, SourceLocation ColonLoc, SourceLocation EndLoc,
CXXScopeSpec &ReductionIdScopeSpec,
const DeclarationNameInfo &ReductionId,
ArrayRef<Expr *> UnresolvedReductions = llvm::None);
OMPClause *ActOnOpenMPInReductionClause(
ArrayRef<Expr *> VarList, SourceLocation StartLoc,
SourceLocation LParenLoc, SourceLocation ColonLoc, SourceLocation EndLoc,
CXXScopeSpec &ReductionIdScopeSpec,
const DeclarationNameInfo &ReductionId,
ArrayRef<Expr *> UnresolvedReductions = llvm::None);
OMPClause *
ActOnOpenMPLinearClause(ArrayRef<Expr *> VarList, Expr *Step,
SourceLocation StartLoc, SourceLocation LParenLoc,
OpenMPLinearClauseKind LinKind, SourceLocation LinLoc,
SourceLocation ColonLoc, SourceLocation EndLoc);
OMPClause *ActOnOpenMPAlignedClause(ArrayRef<Expr *> VarList,
Expr *Alignment,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation ColonLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPCopyinClause(ArrayRef<Expr *> VarList,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPCopyprivateClause(ArrayRef<Expr *> VarList,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPFlushClause(ArrayRef<Expr *> VarList,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *
ActOnOpenMPDependClause(OpenMPDependClauseKind DepKind, SourceLocation DepLoc,
SourceLocation ColonLoc, ArrayRef<Expr *> VarList,
SourceLocation StartLoc, SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPDeviceClause(Expr *Device, SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *
ActOnOpenMPMapClause(OpenMPMapClauseKind MapTypeModifier,
OpenMPMapClauseKind MapType, bool IsMapTypeImplicit,
SourceLocation MapLoc, SourceLocation ColonLoc,
ArrayRef<Expr *> VarList, SourceLocation StartLoc,
SourceLocation LParenLoc, SourceLocation EndLoc);
OMPClause *ActOnOpenMPNumTeamsClause(Expr *NumTeams, SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPThreadLimitClause(Expr *ThreadLimit,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPPriorityClause(Expr *Priority, SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPDistScheduleClause(
OpenMPDistScheduleClauseKind Kind, Expr *ChunkSize,
SourceLocation StartLoc, SourceLocation LParenLoc, SourceLocation KindLoc,
SourceLocation CommaLoc, SourceLocation EndLoc);
OMPClause *ActOnOpenMPDefaultmapClause(
OpenMPDefaultmapClauseModifier M, OpenMPDefaultmapClauseKind Kind,
SourceLocation StartLoc, SourceLocation LParenLoc, SourceLocation MLoc,
SourceLocation KindLoc, SourceLocation EndLoc);
OMPClause *ActOnOpenMPToClause(ArrayRef<Expr *> VarList,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPFromClause(ArrayRef<Expr *> VarList,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPUseDevicePtrClause(ArrayRef<Expr *> VarList,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);
OMPClause *ActOnOpenMPIsDevicePtrClause(ArrayRef<Expr *> VarList,
SourceLocation StartLoc,
SourceLocation LParenLoc,
SourceLocation EndLoc);

enum CheckedConversionKind {
CCK_ImplicitConversion,
CCK_CStyleCast,
CCK_FunctionalCast,
CCK_OtherCast,
CCK_ForBuiltinOverloadedOp
};

static bool isCast(CheckedConversionKind CCK) {
return CCK == CCK_CStyleCast || CCK == CCK_FunctionalCast ||
CCK == CCK_OtherCast;
}

ExprResult ImpCastExprToType(Expr *E, QualType Type, CastKind CK,
ExprValueKind VK = VK_RValue,
const CXXCastPath *BasePath = nullptr,
CheckedConversionKind CCK
= CCK_ImplicitConversion);

static CastKind ScalarTypeToBooleanCastKind(QualType ScalarTy);

ExprResult IgnoredValueConversions(Expr *E);

ExprResult UsualUnaryConversions(Expr *E);

ExprResult CallExprUnaryConversions(Expr *E);

ExprResult DefaultFunctionArrayConversion(Expr *E, bool Diagnose = true);

ExprResult DefaultFunctionArrayLvalueConversion(Expr *E,
bool Diagnose = true);

ExprResult DefaultLvalueConversion(Expr *E);

ExprResult DefaultArgumentPromotion(Expr *E);

ExprResult TemporaryMaterializationConversion(Expr *E);

enum VariadicCallType {
VariadicFunction,
VariadicBlock,
VariadicMethod,
VariadicConstructor,
VariadicDoesNotApply
};

VariadicCallType getVariadicCallType(FunctionDecl *FDecl,
const FunctionProtoType *Proto,
Expr *Fn);

enum VarArgKind {
VAK_Valid,
VAK_ValidInCXX11,
VAK_Undefined,
VAK_MSVCUndefined,
VAK_Invalid
};

VarArgKind isValidVarArgType(const QualType &Ty);

void checkVariadicArgument(const Expr *E, VariadicCallType CT);

bool hasCStrMethod(const Expr *E);

bool GatherArgumentsForCall(SourceLocation CallLoc, FunctionDecl *FDecl,
const FunctionProtoType *Proto,
unsigned FirstParam, ArrayRef<Expr *> Args,
SmallVectorImpl<Expr *> &AllArgs,
VariadicCallType CallType = VariadicDoesNotApply,
bool AllowExplicit = false,
bool IsListInitialization = false);

ExprResult DefaultVariadicArgumentPromotion(Expr *E, VariadicCallType CT,
FunctionDecl *FDecl);

QualType UsualArithmeticConversions(ExprResult &LHS, ExprResult &RHS,
bool IsCompAssign = false);

enum AssignConvertType {
Compatible,

PointerToInt,

IntToPointer,

FunctionVoidPointer,

IncompatiblePointer,

IncompatiblePointerSign,

CompatiblePointerDiscardsQualifiers,

IncompatiblePointerDiscardsQualifiers,

IncompatibleNestedPointerQualifiers,

IncompatibleVectors,

IntToBlockPointer,

IncompatibleBlockPointer,

IncompatibleObjCQualifiedId,

IncompatibleObjCWeakRef,

Incompatible
};

bool DiagnoseAssignmentResult(AssignConvertType ConvTy,
SourceLocation Loc,
QualType DstType, QualType SrcType,
Expr *SrcExpr, AssignmentAction Action,
bool *Complained = nullptr);

bool IsValueInFlagEnum(const EnumDecl *ED, const llvm::APInt &Val,
bool AllowMask) const;

void DiagnoseAssignmentEnum(QualType DstType, QualType SrcType,
Expr *SrcExpr);

AssignConvertType CheckAssignmentConstraints(SourceLocation Loc,
QualType LHSType,
QualType RHSType);

AssignConvertType CheckAssignmentConstraints(QualType LHSType,
ExprResult &RHS,
CastKind &Kind,
bool ConvertRHS = true);

AssignConvertType CheckSingleAssignmentConstraints(
QualType LHSType, ExprResult &RHS, bool Diagnose = true,
bool DiagnoseCFAudited = false, bool ConvertRHS = true);

AssignConvertType CheckTransparentUnionArgumentConstraints(QualType ArgType,
ExprResult &RHS);

bool IsStringLiteralToNonConstPointerConversion(Expr *From, QualType ToType);

bool CheckExceptionSpecCompatibility(Expr *From, QualType ToType);

ExprResult PerformImplicitConversion(Expr *From, QualType ToType,
AssignmentAction Action,
bool AllowExplicit = false);
ExprResult PerformImplicitConversion(Expr *From, QualType ToType,
AssignmentAction Action,
bool AllowExplicit,
ImplicitConversionSequence& ICS);
ExprResult PerformImplicitConversion(Expr *From, QualType ToType,
const ImplicitConversionSequence& ICS,
AssignmentAction Action,
CheckedConversionKind CCK
= CCK_ImplicitConversion);
ExprResult PerformImplicitConversion(Expr *From, QualType ToType,
const StandardConversionSequence& SCS,
AssignmentAction Action,
CheckedConversionKind CCK);


QualType InvalidOperands(SourceLocation Loc, ExprResult &LHS,
ExprResult &RHS);
QualType InvalidLogicalVectorOperands(SourceLocation Loc, ExprResult &LHS,
ExprResult &RHS);
QualType CheckPointerToMemberOperands( 
ExprResult &LHS, ExprResult &RHS, ExprValueKind &VK,
SourceLocation OpLoc, bool isIndirect);
QualType CheckMultiplyDivideOperands( 
ExprResult &LHS, ExprResult &RHS, SourceLocation Loc, bool IsCompAssign,
bool IsDivide);
QualType CheckRemainderOperands( 
ExprResult &LHS, ExprResult &RHS, SourceLocation Loc,
bool IsCompAssign = false);
QualType CheckAdditionOperands( 
ExprResult &LHS, ExprResult &RHS, SourceLocation Loc,
BinaryOperatorKind Opc, QualType* CompLHSTy = nullptr);
QualType CheckSubtractionOperands( 
ExprResult &LHS, ExprResult &RHS, SourceLocation Loc,
QualType* CompLHSTy = nullptr);
QualType CheckShiftOperands( 
ExprResult &LHS, ExprResult &RHS, SourceLocation Loc,
BinaryOperatorKind Opc, bool IsCompAssign = false);
QualType CheckCompareOperands( 
ExprResult &LHS, ExprResult &RHS, SourceLocation Loc,
BinaryOperatorKind Opc);
QualType CheckBitwiseOperands( 
ExprResult &LHS, ExprResult &RHS, SourceLocation Loc,
BinaryOperatorKind Opc);
QualType CheckLogicalOperands( 
ExprResult &LHS, ExprResult &RHS, SourceLocation Loc,
BinaryOperatorKind Opc);
QualType CheckAssignmentOperands( 
Expr *LHSExpr, ExprResult &RHS, SourceLocation Loc, QualType CompoundType);

ExprResult checkPseudoObjectIncDec(Scope *S, SourceLocation OpLoc,
UnaryOperatorKind Opcode, Expr *Op);
ExprResult checkPseudoObjectAssignment(Scope *S, SourceLocation OpLoc,
BinaryOperatorKind Opcode,
Expr *LHS, Expr *RHS);
ExprResult checkPseudoObjectRValue(Expr *E);
Expr *recreateSyntacticForm(PseudoObjectExpr *E);

QualType CheckConditionalOperands( 
ExprResult &Cond, ExprResult &LHS, ExprResult &RHS,
ExprValueKind &VK, ExprObjectKind &OK, SourceLocation QuestionLoc);
QualType CXXCheckConditionalOperands( 
ExprResult &cond, ExprResult &lhs, ExprResult &rhs,
ExprValueKind &VK, ExprObjectKind &OK, SourceLocation questionLoc);
QualType FindCompositePointerType(SourceLocation Loc, Expr *&E1, Expr *&E2,
bool ConvertArgs = true);
QualType FindCompositePointerType(SourceLocation Loc,
ExprResult &E1, ExprResult &E2,
bool ConvertArgs = true) {
Expr *E1Tmp = E1.get(), *E2Tmp = E2.get();
QualType Composite =
FindCompositePointerType(Loc, E1Tmp, E2Tmp, ConvertArgs);
E1 = E1Tmp;
E2 = E2Tmp;
return Composite;
}

QualType FindCompositeObjCPointerType(ExprResult &LHS, ExprResult &RHS,
SourceLocation QuestionLoc);

bool DiagnoseConditionalForNull(Expr *LHSExpr, Expr *RHSExpr,
SourceLocation QuestionLoc);

void DiagnoseAlwaysNonNullPointer(Expr *E,
Expr::NullPointerConstantKind NullType,
bool IsEqual, SourceRange Range);

QualType CheckVectorOperands(ExprResult &LHS, ExprResult &RHS,
SourceLocation Loc, bool IsCompAssign,
bool AllowBothBool, bool AllowBoolConversion);
QualType GetSignedVectorType(QualType V);
QualType CheckVectorCompareOperands(ExprResult &LHS, ExprResult &RHS,
SourceLocation Loc,
BinaryOperatorKind Opc);
QualType CheckVectorLogicalOperands(ExprResult &LHS, ExprResult &RHS,
SourceLocation Loc);

bool areLaxCompatibleVectorTypes(QualType srcType, QualType destType);
bool isLaxVectorConversion(QualType srcType, QualType destType);

bool CheckForConstantInitializer(Expr *e, QualType t);


enum ReferenceCompareResult {
Ref_Incompatible = 0,
Ref_Related,
Ref_Compatible
};

ReferenceCompareResult CompareReferenceRelationship(SourceLocation Loc,
QualType T1, QualType T2,
bool &DerivedToBase,
bool &ObjCConversion,
bool &ObjCLifetimeConversion);

ExprResult checkUnknownAnyCast(SourceRange TypeRange, QualType CastType,
Expr *CastExpr, CastKind &CastKind,
ExprValueKind &VK, CXXCastPath &Path);

ExprResult forceUnknownAnyToType(Expr *E, QualType ToType);

ExprResult checkUnknownAnyArg(SourceLocation callLoc,
Expr *result, QualType &paramType);

bool CheckVectorCast(SourceRange R, QualType VectorTy, QualType Ty,
CastKind &Kind);

ExprResult prepareVectorSplat(QualType VectorTy, Expr *SplattedExpr);

ExprResult CheckExtVectorCast(SourceRange R, QualType DestTy, Expr *CastExpr,
CastKind &Kind);

ExprResult BuildCXXFunctionalCastExpr(TypeSourceInfo *TInfo, QualType Type,
SourceLocation LParenLoc,
Expr *CastExpr,
SourceLocation RParenLoc);

enum ARCConversionResult { ACR_okay, ACR_unbridged, ACR_error };

ARCConversionResult CheckObjCConversion(SourceRange castRange,
QualType castType, Expr *&op,
CheckedConversionKind CCK,
bool Diagnose = true,
bool DiagnoseCFAudited = false,
BinaryOperatorKind Opc = BO_PtrMemD
);

Expr *stripARCUnbridgedCast(Expr *e);
void diagnoseARCUnbridgedCast(Expr *e);

bool CheckObjCARCUnavailableWeakConversion(QualType castType,
QualType ExprType);

void checkRetainCycles(ObjCMessageExpr *msg);
void checkRetainCycles(Expr *receiver, Expr *argument);
void checkRetainCycles(VarDecl *Var, Expr *Init);

bool checkUnsafeAssigns(SourceLocation Loc, QualType LHS, Expr *RHS);

void checkUnsafeExprAssigns(SourceLocation Loc, Expr *LHS, Expr *RHS);

bool CheckMessageArgumentTypes(QualType ReceiverType,
MultiExprArg Args, Selector Sel,
ArrayRef<SourceLocation> SelectorLocs,
ObjCMethodDecl *Method, bool isClassMessage,
bool isSuperMessage,
SourceLocation lbrac, SourceLocation rbrac,
SourceRange RecRange,
QualType &ReturnType, ExprValueKind &VK);

QualType getMessageSendResultType(QualType ReceiverType,
ObjCMethodDecl *Method,
bool isClassMessage, bool isSuperMessage);

void EmitRelatedResultTypeNote(const Expr *E);

void EmitRelatedResultTypeNoteForReturn(QualType destType);

class ConditionResult {
Decl *ConditionVar;
FullExprArg Condition;
bool Invalid;
bool HasKnownValue;
bool KnownValue;

friend class Sema;
ConditionResult(Sema &S, Decl *ConditionVar, FullExprArg Condition,
bool IsConstexpr)
: ConditionVar(ConditionVar), Condition(Condition), Invalid(false),
HasKnownValue(IsConstexpr && Condition.get() &&
!Condition.get()->isValueDependent()),
KnownValue(HasKnownValue &&
!!Condition.get()->EvaluateKnownConstInt(S.Context)) {}
explicit ConditionResult(bool Invalid)
: ConditionVar(nullptr), Condition(nullptr), Invalid(Invalid),
HasKnownValue(false), KnownValue(false) {}

public:
ConditionResult() : ConditionResult(false) {}
bool isInvalid() const { return Invalid; }
std::pair<VarDecl *, Expr *> get() const {
return std::make_pair(cast_or_null<VarDecl>(ConditionVar),
Condition.get());
}
llvm::Optional<bool> getKnownValue() const {
if (!HasKnownValue)
return None;
return KnownValue;
}
};
static ConditionResult ConditionError() { return ConditionResult(true); }

enum class ConditionKind {
Boolean,     
ConstexprIf, 
Switch       
};

ConditionResult ActOnCondition(Scope *S, SourceLocation Loc,
Expr *SubExpr, ConditionKind CK);

ConditionResult ActOnConditionVariable(Decl *ConditionVar,
SourceLocation StmtLoc,
ConditionKind CK);

DeclResult ActOnCXXConditionDeclaration(Scope *S, Declarator &D);

ExprResult CheckConditionVariable(VarDecl *ConditionVar,
SourceLocation StmtLoc,
ConditionKind CK);
ExprResult CheckSwitchCondition(SourceLocation SwitchLoc, Expr *Cond);

ExprResult CheckBooleanCondition(SourceLocation Loc, Expr *E,
bool IsConstexpr = false);

void DiagnoseAssignmentAsCondition(Expr *E);

void DiagnoseEqualityWithExtraParens(ParenExpr *ParenE);

ExprResult CheckCXXBooleanCondition(Expr *CondExpr, bool IsConstexpr = false);

void ConvertIntegerToTypeWarnOnOverflow(llvm::APSInt &OldVal,
unsigned NewWidth, bool NewSign,
SourceLocation Loc, unsigned DiagID);

bool CheckObjCDeclScope(Decl *D);

class VerifyICEDiagnoser {
public:
bool Suppress;

VerifyICEDiagnoser(bool Suppress = false) : Suppress(Suppress) { }

virtual void diagnoseNotICE(Sema &S, SourceLocation Loc, SourceRange SR) =0;
virtual void diagnoseFold(Sema &S, SourceLocation Loc, SourceRange SR);
virtual ~VerifyICEDiagnoser() { }
};

ExprResult VerifyIntegerConstantExpression(Expr *E, llvm::APSInt *Result,
VerifyICEDiagnoser &Diagnoser,
bool AllowFold = true);
ExprResult VerifyIntegerConstantExpression(Expr *E, llvm::APSInt *Result,
unsigned DiagID,
bool AllowFold = true);
ExprResult VerifyIntegerConstantExpression(Expr *E,
llvm::APSInt *Result = nullptr);

ExprResult VerifyBitField(SourceLocation FieldLoc, IdentifierInfo *FieldName,
QualType FieldTy, bool IsMsStruct,
Expr *BitWidth, bool *ZeroWidth = nullptr);

private:
unsigned ForceCUDAHostDeviceDepth = 0;

public:
void PushForceCUDAHostDevice();

bool PopForceCUDAHostDevice();

llvm::DenseMap<CanonicalDeclPtr<FunctionDecl>,
std::vector<PartialDiagnosticAt>>
CUDADeferredDiags;

struct FunctionDeclAndLoc {
CanonicalDeclPtr<FunctionDecl> FD;
SourceLocation Loc;
};

llvm::DenseSet<FunctionDeclAndLoc> LocsWithCUDACallDiags;

llvm::DenseMap< CanonicalDeclPtr<FunctionDecl>,
FunctionDeclAndLoc>
CUDAKnownEmittedFns;

llvm::DenseMap< CanonicalDeclPtr<FunctionDecl>,
llvm::MapVector<CanonicalDeclPtr<FunctionDecl>,
SourceLocation>>
CUDACallGraph;

class CUDADiagBuilder {
public:
enum Kind {
K_Nop,
K_Immediate,
K_ImmediateWithCallStack,
K_Deferred
};

CUDADiagBuilder(Kind K, SourceLocation Loc, unsigned DiagID,
FunctionDecl *Fn, Sema &S);
~CUDADiagBuilder();

operator bool() const { return ImmediateDiag.hasValue(); }

template <typename T>
friend const CUDADiagBuilder &operator<<(const CUDADiagBuilder &Diag,
const T &Value) {
if (Diag.ImmediateDiag.hasValue())
*Diag.ImmediateDiag << Value;
else if (Diag.PartialDiag.hasValue())
*Diag.PartialDiag << Value;
return Diag;
}

private:
Sema &S;
SourceLocation Loc;
unsigned DiagID;
FunctionDecl *Fn;
bool ShowCallStack;

llvm::Optional<SemaDiagnosticBuilder> ImmediateDiag;
llvm::Optional<PartialDiagnostic> PartialDiag;
};

CUDADiagBuilder CUDADiagIfDeviceCode(SourceLocation Loc, unsigned DiagID);

CUDADiagBuilder CUDADiagIfHostCode(SourceLocation Loc, unsigned DiagID);

enum CUDAFunctionTarget {
CFT_Device,
CFT_Global,
CFT_Host,
CFT_HostDevice,
CFT_InvalidTarget
};

CUDAFunctionTarget IdentifyCUDATarget(const FunctionDecl *D,
bool IgnoreImplicitHDAttr = false);
CUDAFunctionTarget IdentifyCUDATarget(const ParsedAttributesView &Attrs);

CUDAFunctionTarget CurrentCUDATarget() {
return IdentifyCUDATarget(dyn_cast<FunctionDecl>(CurContext));
}

enum CUDAFunctionPreference {
CFP_Never,      
CFP_WrongSide,  
CFP_HostDevice, 
CFP_SameSide,   
CFP_Native,     
};

CUDAFunctionPreference IdentifyCUDAPreference(const FunctionDecl *Caller,
const FunctionDecl *Callee);

bool IsAllowedCUDACall(const FunctionDecl *Caller,
const FunctionDecl *Callee) {
return IdentifyCUDAPreference(Caller, Callee) != CFP_Never;
}

void maybeAddCUDAHostDeviceAttrs(FunctionDecl *FD,
const LookupResult &Previous);

public:
bool CheckCUDACall(SourceLocation Loc, FunctionDecl *Callee);

void CUDASetLambdaAttrs(CXXMethodDecl *Method);

void EraseUnwantedCUDAMatches(
const FunctionDecl *Caller,
SmallVectorImpl<std::pair<DeclAccessPair, FunctionDecl *>> &Matches);

bool inferCUDATargetForImplicitSpecialMember(CXXRecordDecl *ClassDecl,
CXXSpecialMember CSM,
CXXMethodDecl *MemberDecl,
bool ConstRHS,
bool Diagnose);

bool isEmptyCudaConstructor(SourceLocation Loc, CXXConstructorDecl *CD);
bool isEmptyCudaDestructor(SourceLocation Loc, CXXDestructorDecl *CD);

void checkAllowedCUDAInitializer(VarDecl *VD);

void checkCUDATargetOverload(FunctionDecl *NewFD,
const LookupResult &Previous);
void inheritCUDATargetAttrs(FunctionDecl *FD, const FunctionTemplateDecl &TD);

enum ParserCompletionContext {
PCC_Namespace,
PCC_Class,
PCC_ObjCInterface,
PCC_ObjCImplementation,
PCC_ObjCInstanceVariableList,
PCC_Template,
PCC_MemberTemplate,
PCC_Expression,
PCC_Statement,
PCC_ForInit,
PCC_Condition,
PCC_RecoveryInFunction,
PCC_Type,
PCC_ParenthesizedExpression,
PCC_LocalDeclarationSpecifiers
};

void CodeCompleteModuleImport(SourceLocation ImportLoc, ModuleIdPath Path);
void CodeCompleteOrdinaryName(Scope *S,
ParserCompletionContext CompletionContext);
void CodeCompleteDeclSpec(Scope *S, DeclSpec &DS,
bool AllowNonIdentifiers,
bool AllowNestedNameSpecifiers);

struct CodeCompleteExpressionData;
void CodeCompleteExpression(Scope *S,
const CodeCompleteExpressionData &Data);
void CodeCompleteMemberReferenceExpr(Scope *S, Expr *Base, Expr *OtherOpBase,
SourceLocation OpLoc, bool IsArrow,
bool IsBaseExprStatement);
void CodeCompletePostfixExpression(Scope *S, ExprResult LHS);
void CodeCompleteTag(Scope *S, unsigned TagSpec);
void CodeCompleteTypeQualifiers(DeclSpec &DS);
void CodeCompleteFunctionQualifiers(DeclSpec &DS, Declarator &D,
const VirtSpecifiers *VS = nullptr);
void CodeCompleteBracketDeclarator(Scope *S);
void CodeCompleteCase(Scope *S);
void CodeCompleteCall(Scope *S, Expr *Fn, ArrayRef<Expr *> Args);
void CodeCompleteConstructor(Scope *S, QualType Type, SourceLocation Loc,
ArrayRef<Expr *> Args);
void CodeCompleteInitializer(Scope *S, Decl *D);
void CodeCompleteReturn(Scope *S);
void CodeCompleteAfterIf(Scope *S);
void CodeCompleteAssignmentRHS(Scope *S, Expr *LHS);

void CodeCompleteQualifiedId(Scope *S, CXXScopeSpec &SS,
bool EnteringContext);
void CodeCompleteUsing(Scope *S);
void CodeCompleteUsingDirective(Scope *S);
void CodeCompleteNamespaceDecl(Scope *S);
void CodeCompleteNamespaceAliasDecl(Scope *S);
void CodeCompleteOperatorName(Scope *S);
void CodeCompleteConstructorInitializer(
Decl *Constructor,
ArrayRef<CXXCtorInitializer *> Initializers);

void CodeCompleteLambdaIntroducer(Scope *S, LambdaIntroducer &Intro,
bool AfterAmpersand);

void CodeCompleteObjCAtDirective(Scope *S);
void CodeCompleteObjCAtVisibility(Scope *S);
void CodeCompleteObjCAtStatement(Scope *S);
void CodeCompleteObjCAtExpression(Scope *S);
void CodeCompleteObjCPropertyFlags(Scope *S, ObjCDeclSpec &ODS);
void CodeCompleteObjCPropertyGetter(Scope *S);
void CodeCompleteObjCPropertySetter(Scope *S);
void CodeCompleteObjCPassingType(Scope *S, ObjCDeclSpec &DS,
bool IsParameter);
void CodeCompleteObjCMessageReceiver(Scope *S);
void CodeCompleteObjCSuperMessage(Scope *S, SourceLocation SuperLoc,
ArrayRef<IdentifierInfo *> SelIdents,
bool AtArgumentExpression);
void CodeCompleteObjCClassMessage(Scope *S, ParsedType Receiver,
ArrayRef<IdentifierInfo *> SelIdents,
bool AtArgumentExpression,
bool IsSuper = false);
void CodeCompleteObjCInstanceMessage(Scope *S, Expr *Receiver,
ArrayRef<IdentifierInfo *> SelIdents,
bool AtArgumentExpression,
ObjCInterfaceDecl *Super = nullptr);
void CodeCompleteObjCForCollection(Scope *S,
DeclGroupPtrTy IterationVar);
void CodeCompleteObjCSelector(Scope *S,
ArrayRef<IdentifierInfo *> SelIdents);
void CodeCompleteObjCProtocolReferences(
ArrayRef<IdentifierLocPair> Protocols);
void CodeCompleteObjCProtocolDecl(Scope *S);
void CodeCompleteObjCInterfaceDecl(Scope *S);
void CodeCompleteObjCSuperclass(Scope *S,
IdentifierInfo *ClassName,
SourceLocation ClassNameLoc);
void CodeCompleteObjCImplementationDecl(Scope *S);
void CodeCompleteObjCInterfaceCategory(Scope *S,
IdentifierInfo *ClassName,
SourceLocation ClassNameLoc);
void CodeCompleteObjCImplementationCategory(Scope *S,
IdentifierInfo *ClassName,
SourceLocation ClassNameLoc);
void CodeCompleteObjCPropertyDefinition(Scope *S);
void CodeCompleteObjCPropertySynthesizeIvar(Scope *S,
IdentifierInfo *PropertyName);
void CodeCompleteObjCMethodDecl(Scope *S, Optional<bool> IsInstanceMethod,
ParsedType ReturnType);
void CodeCompleteObjCMethodDeclSelector(Scope *S,
bool IsInstanceMethod,
bool AtParameterName,
ParsedType ReturnType,
ArrayRef<IdentifierInfo *> SelIdents);
void CodeCompleteObjCClassPropertyRefExpr(Scope *S, IdentifierInfo &ClassName,
SourceLocation ClassNameLoc,
bool IsBaseExprStatement);
void CodeCompletePreprocessorDirective(bool InConditional);
void CodeCompleteInPreprocessorConditionalExclusion(Scope *S);
void CodeCompletePreprocessorMacroName(bool IsDefinition);
void CodeCompletePreprocessorExpression();
void CodeCompletePreprocessorMacroArgument(Scope *S,
IdentifierInfo *Macro,
MacroInfo *MacroInfo,
unsigned Argument);
void CodeCompleteNaturalLanguage();
void CodeCompleteAvailabilityPlatformName();
void GatherGlobalCodeCompletions(CodeCompletionAllocator &Allocator,
CodeCompletionTUInfo &CCTUInfo,
SmallVectorImpl<CodeCompletionResult> &Results);


public:
SourceLocation getLocationOfStringLiteralByte(const StringLiteral *SL,
unsigned ByteNo) const;

private:
void CheckArrayAccess(const Expr *BaseExpr, const Expr *IndexExpr,
const ArraySubscriptExpr *ASE=nullptr,
bool AllowOnePastEnd=true, bool IndexNegated=false);
void CheckArrayAccess(const Expr *E);
struct FormatStringInfo {
unsigned FormatIdx;
unsigned FirstDataArg;
bool HasVAListArg;
};

static bool getFormatStringInfo(const FormatAttr *Format, bool IsCXXMember,
FormatStringInfo *FSI);
bool CheckFunctionCall(FunctionDecl *FDecl, CallExpr *TheCall,
const FunctionProtoType *Proto);
bool CheckObjCMethodCall(ObjCMethodDecl *Method, SourceLocation loc,
ArrayRef<const Expr *> Args);
bool CheckPointerCall(NamedDecl *NDecl, CallExpr *TheCall,
const FunctionProtoType *Proto);
bool CheckOtherCall(CallExpr *TheCall, const FunctionProtoType *Proto);
void CheckConstructorCall(FunctionDecl *FDecl,
ArrayRef<const Expr *> Args,
const FunctionProtoType *Proto,
SourceLocation Loc);

void checkCall(NamedDecl *FDecl, const FunctionProtoType *Proto,
const Expr *ThisArg, ArrayRef<const Expr *> Args,
bool IsMemberFunction, SourceLocation Loc, SourceRange Range,
VariadicCallType CallType);

bool CheckObjCString(Expr *Arg);
ExprResult CheckOSLogFormatStringArg(Expr *Arg);

ExprResult CheckBuiltinFunctionCall(FunctionDecl *FDecl,
unsigned BuiltinID, CallExpr *TheCall);

bool CheckARMBuiltinExclusiveCall(unsigned BuiltinID, CallExpr *TheCall,
unsigned MaxWidth);
bool CheckNeonBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
bool CheckARMBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);

bool CheckAArch64BuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
bool CheckHexagonBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
bool CheckHexagonBuiltinCpu(unsigned BuiltinID, CallExpr *TheCall);
bool CheckHexagonBuiltinArgument(unsigned BuiltinID, CallExpr *TheCall);
bool CheckMipsBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
bool CheckSystemZBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
bool CheckX86BuiltinRoundingOrSAE(unsigned BuiltinID, CallExpr *TheCall);
bool CheckX86BuiltinGatherScatterScale(unsigned BuiltinID, CallExpr *TheCall);
bool CheckX86BuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
bool CheckPPCBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);

bool SemaBuiltinVAStart(unsigned BuiltinID, CallExpr *TheCall);
bool SemaBuiltinVAStartARMMicrosoft(CallExpr *Call);
bool SemaBuiltinUnorderedCompare(CallExpr *TheCall);
bool SemaBuiltinFPClassification(CallExpr *TheCall, unsigned NumArgs);
bool SemaBuiltinVSX(CallExpr *TheCall);
bool SemaBuiltinOSLogFormat(CallExpr *TheCall);

public:
ExprResult SemaBuiltinShuffleVector(CallExpr *TheCall);
ExprResult SemaConvertVectorExpr(Expr *E, TypeSourceInfo *TInfo,
SourceLocation BuiltinLoc,
SourceLocation RParenLoc);

private:
bool SemaBuiltinPrefetch(CallExpr *TheCall);
bool SemaBuiltinAllocaWithAlign(CallExpr *TheCall);
bool SemaBuiltinAssume(CallExpr *TheCall);
bool SemaBuiltinAssumeAligned(CallExpr *TheCall);
bool SemaBuiltinLongjmp(CallExpr *TheCall);
bool SemaBuiltinSetjmp(CallExpr *TheCall);
ExprResult SemaBuiltinAtomicOverloaded(ExprResult TheCallResult);
ExprResult SemaBuiltinNontemporalOverloaded(ExprResult TheCallResult);
ExprResult SemaAtomicOpsOverloaded(ExprResult TheCallResult,
AtomicExpr::AtomicOp Op);
ExprResult SemaBuiltinOperatorNewDeleteOverloaded(ExprResult TheCallResult,
bool IsDelete);
bool SemaBuiltinConstantArg(CallExpr *TheCall, int ArgNum,
llvm::APSInt &Result);
bool SemaBuiltinConstantArgRange(CallExpr *TheCall, int ArgNum, int Low,
int High, bool RangeIsError = true);
bool SemaBuiltinConstantArgMultiple(CallExpr *TheCall, int ArgNum,
unsigned Multiple);
bool SemaBuiltinARMSpecialReg(unsigned BuiltinID, CallExpr *TheCall,
int ArgNum, unsigned ExpectedFieldNum,
bool AllowName);
public:
enum FormatStringType {
FST_Scanf,
FST_Printf,
FST_NSString,
FST_Strftime,
FST_Strfmon,
FST_Kprintf,
FST_FreeBSDKPrintf,
FST_OSTrace,
FST_OSLog,
FST_Unknown
};
static FormatStringType GetFormatStringType(const FormatAttr *Format);

bool FormatStringHasSArg(const StringLiteral *FExpr);

static bool GetFormatNSStringIdx(const FormatAttr *Format, unsigned &Idx);

private:
bool CheckFormatArguments(const FormatAttr *Format,
ArrayRef<const Expr *> Args,
bool IsCXXMember,
VariadicCallType CallType,
SourceLocation Loc, SourceRange Range,
llvm::SmallBitVector &CheckedVarArgs);
bool CheckFormatArguments(ArrayRef<const Expr *> Args,
bool HasVAListArg, unsigned format_idx,
unsigned firstDataArg, FormatStringType Type,
VariadicCallType CallType,
SourceLocation Loc, SourceRange range,
llvm::SmallBitVector &CheckedVarArgs);

void CheckAbsoluteValueFunction(const CallExpr *Call,
const FunctionDecl *FDecl);

void CheckMaxUnsignedZero(const CallExpr *Call, const FunctionDecl *FDecl);

void CheckMemaccessArguments(const CallExpr *Call,
unsigned BId,
IdentifierInfo *FnName);

void CheckStrlcpycatArguments(const CallExpr *Call,
IdentifierInfo *FnName);

void CheckStrncatArguments(const CallExpr *Call,
IdentifierInfo *FnName);

void CheckReturnValExpr(Expr *RetValExp, QualType lhsType,
SourceLocation ReturnLoc,
bool isObjCMethod = false,
const AttrVec *Attrs = nullptr,
const FunctionDecl *FD = nullptr);

public:
void CheckFloatComparison(SourceLocation Loc, Expr *LHS, Expr *RHS);

private:
void CheckImplicitConversions(Expr *E, SourceLocation CC = SourceLocation());
void CheckBoolLikeConversion(Expr *E, SourceLocation CC);
void CheckForIntOverflow(Expr *E);
void CheckUnsequencedOperations(Expr *E);

void CheckCompletedExpr(Expr *E, SourceLocation CheckLoc = SourceLocation(),
bool IsConstexpr = false);

void CheckBitFieldInitialization(SourceLocation InitLoc, FieldDecl *Field,
Expr *Init);

void CheckShadowInheritedFields(const SourceLocation &Loc,
DeclarationName FieldName,
const CXXRecordDecl *RD);

void CheckBreakContinueBinding(Expr *E);

void CheckObjCCircularContainer(ObjCMessageExpr *Message);

void AnalyzeDeleteExprMismatch(const CXXDeleteExpr *DE);
void AnalyzeDeleteExprMismatch(FieldDecl *Field, SourceLocation DeleteLoc,
bool DeleteWasArrayForm);
public:
void RegisterTypeTagForDatatype(const IdentifierInfo *ArgumentKind,
uint64_t MagicValue, QualType Type,
bool LayoutCompatible, bool MustBeNull);

struct TypeTagData {
TypeTagData() {}

TypeTagData(QualType Type, bool LayoutCompatible, bool MustBeNull) :
Type(Type), LayoutCompatible(LayoutCompatible),
MustBeNull(MustBeNull)
{}

QualType Type;

unsigned LayoutCompatible : 1;
unsigned MustBeNull : 1;
};

typedef std::pair<const IdentifierInfo *, uint64_t> TypeTagMagicValue;

private:
std::unique_ptr<llvm::DenseMap<TypeTagMagicValue, TypeTagData>>
TypeTagForDatatypeMagicValues;

void CheckArgumentWithTypeTag(const ArgumentWithTypeTagAttr *Attr,
const ArrayRef<const Expr *> ExprArgs,
SourceLocation CallSiteLoc);

void CheckAddressOfPackedMember(Expr *rhs);

Scope *CurScope;

mutable IdentifierInfo *Ident_super;
mutable IdentifierInfo *Ident___float128;

IdentifierInfo *Ident__Nonnull = nullptr;
IdentifierInfo *Ident__Nullable = nullptr;
IdentifierInfo *Ident__Null_unspecified = nullptr;

IdentifierInfo *Ident_NSError = nullptr;

sema::SemaPPCallbacks *SemaPPCallbackHandler;

protected:
friend class Parser;
friend class InitializationSequence;
friend class ASTReader;
friend class ASTDeclReader;
friend class ASTWriter;

public:
IdentifierInfo *getNullabilityKeyword(NullabilityKind nullability);

RecordDecl *CFError = nullptr;

IdentifierInfo *getNSErrorIdent();

Scope *getCurScope() const { return CurScope; }

void incrementMSManglingNumber() const {
return CurScope->incrementMSManglingNumber();
}

IdentifierInfo *getSuperIdentifier() const;
IdentifierInfo *getFloat128Identifier() const;

Decl *getObjCDeclContext() const;

DeclContext *getCurLexicalContext() const {
return OriginalLexicalContext ? OriginalLexicalContext : CurContext;
}

const DeclContext *getCurObjCLexicalContext() const {
const DeclContext *DC = getCurLexicalContext();
if (const ObjCCategoryDecl *CatD = dyn_cast<ObjCCategoryDecl>(DC))
DC = CatD->getClassInterface();
return DC;
}

static bool TooManyArguments(size_t NumParams, size_t NumArgs,
bool PartialOverloading = false) {
if (NumArgs > 0 && PartialOverloading)
return NumArgs + 1 > NumParams; 
return NumArgs > NumParams;
}

SmallVector<CXXRecordDecl*, 4> DelayedDllExportClasses;

private:
class SavePendingParsedClassStateRAII {
public:
SavePendingParsedClassStateRAII(Sema &S) : S(S) { swapSavedState(); }

~SavePendingParsedClassStateRAII() {
assert(S.DelayedExceptionSpecChecks.empty() &&
"there shouldn't be any pending delayed exception spec checks");
assert(S.DelayedDefaultedMemberExceptionSpecs.empty() &&
"there shouldn't be any pending delayed defaulted member "
"exception specs");
assert(S.DelayedDllExportClasses.empty() &&
"there shouldn't be any pending delayed DLL export classes");
swapSavedState();
}

private:
Sema &S;
decltype(DelayedExceptionSpecChecks) SavedExceptionSpecChecks;
decltype(DelayedDefaultedMemberExceptionSpecs)
SavedDefaultedMemberExceptionSpecs;
decltype(DelayedDllExportClasses) SavedDllExportClasses;

void swapSavedState() {
SavedExceptionSpecChecks.swap(S.DelayedExceptionSpecChecks);
SavedDefaultedMemberExceptionSpecs.swap(
S.DelayedDefaultedMemberExceptionSpecs);
SavedDllExportClasses.swap(S.DelayedDllExportClasses);
}
};

struct MisalignedMember {
Expr *E;
RecordDecl *RD;
ValueDecl *MD;
CharUnits Alignment;

MisalignedMember() : E(), RD(), MD(), Alignment() {}
MisalignedMember(Expr *E, RecordDecl *RD, ValueDecl *MD,
CharUnits Alignment)
: E(E), RD(RD), MD(MD), Alignment(Alignment) {}
explicit MisalignedMember(Expr *E)
: MisalignedMember(E, nullptr, nullptr, CharUnits()) {}

bool operator==(const MisalignedMember &m) { return this->E == m.E; }
};
SmallVector<MisalignedMember, 4> MisalignedMembers;

void AddPotentialMisalignedMembers(Expr *E, RecordDecl *RD, ValueDecl *MD,
CharUnits Alignment);

public:
void DiagnoseMisalignedMembers();

void DiscardMisalignedMemberAddress(const Type *T, Expr *E);

void RefersToMemberWithReducedAlignment(
Expr *E,
llvm::function_ref<void(Expr *, RecordDecl *, FieldDecl *, CharUnits)>
Action);
};

class EnterExpressionEvaluationContext {
Sema &Actions;
bool Entered = true;

public:
EnterExpressionEvaluationContext(
Sema &Actions, Sema::ExpressionEvaluationContext NewContext,
Decl *LambdaContextDecl = nullptr,
Sema::ExpressionEvaluationContextRecord::ExpressionKind ExprContext =
Sema::ExpressionEvaluationContextRecord::EK_Other,
bool ShouldEnter = true)
: Actions(Actions), Entered(ShouldEnter) {
if (Entered)
Actions.PushExpressionEvaluationContext(NewContext, LambdaContextDecl,
ExprContext);
}
EnterExpressionEvaluationContext(
Sema &Actions, Sema::ExpressionEvaluationContext NewContext,
Sema::ReuseLambdaContextDecl_t,
Sema::ExpressionEvaluationContextRecord::ExpressionKind ExprContext =
Sema::ExpressionEvaluationContextRecord::EK_Other)
: Actions(Actions) {
Actions.PushExpressionEvaluationContext(
NewContext, Sema::ReuseLambdaContextDecl, ExprContext);
}

enum InitListTag { InitList };
EnterExpressionEvaluationContext(Sema &Actions, InitListTag,
bool ShouldEnter = true)
: Actions(Actions), Entered(false) {
if (ShouldEnter && Actions.isUnevaluatedContext() &&
Actions.getLangOpts().CPlusPlus11) {
Actions.PushExpressionEvaluationContext(
Sema::ExpressionEvaluationContext::UnevaluatedList);
Entered = true;
}
}

~EnterExpressionEvaluationContext() {
if (Entered)
Actions.PopExpressionEvaluationContext();
}
};

DeductionFailureInfo
MakeDeductionFailureInfo(ASTContext &Context, Sema::TemplateDeductionResult TDK,
sema::TemplateDeductionInfo &Info);

struct LateParsedTemplate {
CachedTokens Toks;
Decl *D;
};

} 

namespace llvm {
template <> struct DenseMapInfo<clang::Sema::FunctionDeclAndLoc> {
using FunctionDeclAndLoc = clang::Sema::FunctionDeclAndLoc;
using FDBaseInfo = DenseMapInfo<clang::CanonicalDeclPtr<clang::FunctionDecl>>;

static FunctionDeclAndLoc getEmptyKey() {
return {FDBaseInfo::getEmptyKey(), clang::SourceLocation()};
}

static FunctionDeclAndLoc getTombstoneKey() {
return {FDBaseInfo::getTombstoneKey(), clang::SourceLocation()};
}

static unsigned getHashValue(const FunctionDeclAndLoc &FDL) {
return hash_combine(FDBaseInfo::getHashValue(FDL.FD),
FDL.Loc.getRawEncoding());
}

static bool isEqual(const FunctionDeclAndLoc &LHS,
const FunctionDeclAndLoc &RHS) {
return LHS.FD == RHS.FD && LHS.Loc == RHS.Loc;
}
};
} 

#endif
