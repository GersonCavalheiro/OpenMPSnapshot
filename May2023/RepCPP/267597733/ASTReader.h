
#ifndef LLVM_CLANG_SERIALIZATION_ASTREADER_H
#define LLVM_CLANG_SERIALIZATION_ASTREADER_H

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/TemplateName.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/OpenCLOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Version.h"
#include "clang/Lex/ExternalPreprocessorSource.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Lex/Token.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/IdentifierResolver.h"
#include "clang/Serialization/ASTBitCodes.h"
#include "clang/Serialization/ContinuousRangeMap.h"
#include "clang/Serialization/Module.h"
#include "clang/Serialization/ModuleFileExtension.h"
#include "clang/Serialization/ModuleManager.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/VersionTuple.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <deque>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace clang {

class ASTConsumer;
class ASTContext;
class ASTDeserializationListener;
class ASTReader;
class ASTRecordReader;
class CXXTemporary;
class Decl;
class DeclaratorDecl;
class DeclContext;
class EnumDecl;
class Expr;
class FieldDecl;
class FileEntry;
class FileManager;
class FileSystemOptions;
class FunctionDecl;
class GlobalModuleIndex;
struct HeaderFileInfo;
class HeaderSearchOptions;
class LangOptions;
class LazyASTUnresolvedSet;
class MacroInfo;
class MemoryBufferCache;
class NamedDecl;
class NamespaceDecl;
class ObjCCategoryDecl;
class ObjCInterfaceDecl;
class PCHContainerReader;
class Preprocessor;
class PreprocessorOptions;
struct QualifierInfo;
class Sema;
class SourceManager;
class Stmt;
class SwitchCase;
class TargetOptions;
class TemplateParameterList;
class TypedefNameDecl;
class TypeSourceInfo;
class ValueDecl;
class VarDecl;

class ASTReaderListener {
public:
virtual ~ASTReaderListener();

virtual bool ReadFullVersionInformation(StringRef FullVersion) {
return FullVersion != getClangFullRepositoryVersion();
}

virtual void ReadModuleName(StringRef ModuleName) {}
virtual void ReadModuleMapFile(StringRef ModuleMapPath) {}

virtual bool ReadLanguageOptions(const LangOptions &LangOpts,
bool Complain,
bool AllowCompatibleDifferences) {
return false;
}

virtual bool ReadTargetOptions(const TargetOptions &TargetOpts, bool Complain,
bool AllowCompatibleDifferences) {
return false;
}

virtual bool
ReadDiagnosticOptions(IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts,
bool Complain) {
return false;
}

virtual bool ReadFileSystemOptions(const FileSystemOptions &FSOpts,
bool Complain) {
return false;
}

virtual bool ReadHeaderSearchOptions(const HeaderSearchOptions &HSOpts,
StringRef SpecificModuleCachePath,
bool Complain) {
return false;
}

virtual bool ReadPreprocessorOptions(const PreprocessorOptions &PPOpts,
bool Complain,
std::string &SuggestedPredefines) {
return false;
}

virtual void ReadCounter(const serialization::ModuleFile &M,
unsigned Value) {}

virtual void visitModuleFile(StringRef Filename,
serialization::ModuleKind Kind) {}

virtual bool needsInputFileVisitation() { return false; }

virtual bool needsSystemInputFileVisitation() { return false; }

virtual bool visitInputFile(StringRef Filename, bool isSystem,
bool isOverridden, bool isExplicitModule) {
return true;
}

virtual bool needsImportVisitation() const { return false; }

virtual void visitImport(StringRef Filename) {}

virtual void readModuleFileExtension(
const ModuleFileExtensionMetadata &Metadata) {}
};

class ChainedASTReaderListener : public ASTReaderListener {
std::unique_ptr<ASTReaderListener> First;
std::unique_ptr<ASTReaderListener> Second;

public:
ChainedASTReaderListener(std::unique_ptr<ASTReaderListener> First,
std::unique_ptr<ASTReaderListener> Second)
: First(std::move(First)), Second(std::move(Second)) {}

std::unique_ptr<ASTReaderListener> takeFirst() { return std::move(First); }
std::unique_ptr<ASTReaderListener> takeSecond() { return std::move(Second); }

bool ReadFullVersionInformation(StringRef FullVersion) override;
void ReadModuleName(StringRef ModuleName) override;
void ReadModuleMapFile(StringRef ModuleMapPath) override;
bool ReadLanguageOptions(const LangOptions &LangOpts, bool Complain,
bool AllowCompatibleDifferences) override;
bool ReadTargetOptions(const TargetOptions &TargetOpts, bool Complain,
bool AllowCompatibleDifferences) override;
bool ReadDiagnosticOptions(IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts,
bool Complain) override;
bool ReadFileSystemOptions(const FileSystemOptions &FSOpts,
bool Complain) override;

bool ReadHeaderSearchOptions(const HeaderSearchOptions &HSOpts,
StringRef SpecificModuleCachePath,
bool Complain) override;
bool ReadPreprocessorOptions(const PreprocessorOptions &PPOpts,
bool Complain,
std::string &SuggestedPredefines) override;

void ReadCounter(const serialization::ModuleFile &M, unsigned Value) override;
bool needsInputFileVisitation() override;
bool needsSystemInputFileVisitation() override;
void visitModuleFile(StringRef Filename,
serialization::ModuleKind Kind) override;
bool visitInputFile(StringRef Filename, bool isSystem,
bool isOverridden, bool isExplicitModule) override;
void readModuleFileExtension(
const ModuleFileExtensionMetadata &Metadata) override;
};

class PCHValidator : public ASTReaderListener {
Preprocessor &PP;
ASTReader &Reader;

public:
PCHValidator(Preprocessor &PP, ASTReader &Reader)
: PP(PP), Reader(Reader) {}

bool ReadLanguageOptions(const LangOptions &LangOpts, bool Complain,
bool AllowCompatibleDifferences) override;
bool ReadTargetOptions(const TargetOptions &TargetOpts, bool Complain,
bool AllowCompatibleDifferences) override;
bool ReadDiagnosticOptions(IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts,
bool Complain) override;
bool ReadPreprocessorOptions(const PreprocessorOptions &PPOpts, bool Complain,
std::string &SuggestedPredefines) override;
bool ReadHeaderSearchOptions(const HeaderSearchOptions &HSOpts,
StringRef SpecificModuleCachePath,
bool Complain) override;
void ReadCounter(const serialization::ModuleFile &M, unsigned Value) override;

private:
void Error(const char *Msg);
};

class SimpleASTReaderListener : public ASTReaderListener {
Preprocessor &PP;

public:
SimpleASTReaderListener(Preprocessor &PP) : PP(PP) {}

bool ReadPreprocessorOptions(const PreprocessorOptions &PPOpts, bool Complain,
std::string &SuggestedPredefines) override;
};

namespace serialization {

class ReadMethodPoolVisitor;

namespace reader {

class ASTIdentifierLookupTrait;

struct DeclContextLookupTable;

} 

} 

class ASTReader
: public ExternalPreprocessorSource,
public ExternalPreprocessingRecordSource,
public ExternalHeaderFileInfoSource,
public ExternalSemaSource,
public IdentifierInfoLookup,
public ExternalSLocEntrySource
{
public:
friend class ASTDeclReader;
friend class ASTIdentifierIterator;
friend class ASTRecordReader;
friend class ASTStmtReader;
friend class ASTUnit; 
friend class ASTWriter;
friend class PCHValidator;
friend class serialization::reader::ASTIdentifierLookupTrait;
friend class serialization::ReadMethodPoolVisitor;
friend class TypeLocReader;

using RecordData = SmallVector<uint64_t, 64>;
using RecordDataImpl = SmallVectorImpl<uint64_t>;

enum ASTReadResult {
Success,

Failure,

Missing,

OutOfDate,

VersionMismatch,

ConfigurationMismatch,

HadErrors
};

using ModuleFile = serialization::ModuleFile;
using ModuleKind = serialization::ModuleKind;
using ModuleManager = serialization::ModuleManager;
using ModuleIterator = ModuleManager::ModuleIterator;
using ModuleConstIterator = ModuleManager::ModuleConstIterator;
using ModuleReverseIterator = ModuleManager::ModuleReverseIterator;

private:
std::unique_ptr<ASTReaderListener> Listener;

ASTDeserializationListener *DeserializationListener = nullptr;

bool OwnsDeserializationListener = false;

SourceManager &SourceMgr;
FileManager &FileMgr;
const PCHContainerReader &PCHContainerRdr;
DiagnosticsEngine &Diags;

Sema *SemaObj = nullptr;

Preprocessor &PP;

ASTContext *ContextObj = nullptr;

ASTConsumer *Consumer = nullptr;

ModuleManager ModuleMgr;

MemoryBufferCache &PCMCache;

IdentifierResolver DummyIdResolver;

llvm::StringMap<std::shared_ptr<ModuleFileExtension>> ModuleFileExtensions;

std::unique_ptr<llvm::Timer> ReadTimer;

SourceLocation CurrentImportLoc;

std::unique_ptr<GlobalModuleIndex> GlobalIndex;

ContinuousRangeMap<uint64_t, ModuleFile*, 4> GlobalBitOffsetsMap;

ContinuousRangeMap<unsigned, ModuleFile*, 64> GlobalSLocEntryMap;

using GlobalSLocOffsetMapType =
ContinuousRangeMap<unsigned, ModuleFile *, 64>;

GlobalSLocOffsetMapType GlobalSLocOffsetMap;

std::vector<QualType> TypesLoaded;

using GlobalTypeMapType =
ContinuousRangeMap<serialization::TypeID, ModuleFile *, 4>;

GlobalTypeMapType GlobalTypeMap;

std::vector<Decl *> DeclsLoaded;

using GlobalDeclMapType =
ContinuousRangeMap<serialization::DeclID, ModuleFile *, 4>;

GlobalDeclMapType GlobalDeclMap;

using FileOffset = std::pair<ModuleFile *, uint64_t>;
using FileOffsetsTy = SmallVector<FileOffset, 2>;
using DeclUpdateOffsetsMap =
llvm::DenseMap<serialization::DeclID, FileOffsetsTy>;

DeclUpdateOffsetsMap DeclUpdateOffsets;

struct PendingUpdateRecord {
Decl *D;
serialization::GlobalDeclID ID;

bool JustLoaded;

PendingUpdateRecord(serialization::GlobalDeclID ID, Decl *D,
bool JustLoaded)
: D(D), ID(ID), JustLoaded(JustLoaded) {}
};

llvm::SmallVector<PendingUpdateRecord, 16> PendingUpdateRecords;

enum class PendingFakeDefinitionKind { NotFake, Fake, FakeLoaded };

llvm::DenseMap<void *, PendingFakeDefinitionKind> PendingFakeDefinitionData;

llvm::SmallMapVector<Decl *, FunctionDecl *, 4> PendingExceptionSpecUpdates;

llvm::DenseMap<std::pair<DeclContext *, IdentifierInfo *>, NamedDecl *>
ImportedTypedefNamesForLinkage;

llvm::DenseMap<Decl*, llvm::SmallVector<NamedDecl*, 2>>
AnonymousDeclarationsForMerging;

struct FileDeclsInfo {
ModuleFile *Mod = nullptr;
ArrayRef<serialization::LocalDeclID> Decls;

FileDeclsInfo() = default;
FileDeclsInfo(ModuleFile *Mod, ArrayRef<serialization::LocalDeclID> Decls)
: Mod(Mod), Decls(Decls) {}
};

llvm::DenseMap<FileID, FileDeclsInfo> FileDeclIDs;

using LexicalContents = ArrayRef<llvm::support::unaligned_uint32_t>;

llvm::DenseMap<const DeclContext*, std::pair<ModuleFile*, LexicalContents>>
LexicalDecls;

std::vector<std::pair<ModuleFile*, LexicalContents>> TULexicalDecls;

llvm::DenseMap<const DeclContext *,
serialization::reader::DeclContextLookupTable> Lookups;

struct PendingVisibleUpdate {
ModuleFile *Mod;
const unsigned char *Data;
};
using DeclContextVisibleUpdates = SmallVector<PendingVisibleUpdate, 1>;

llvm::DenseMap<serialization::DeclID, DeclContextVisibleUpdates>
PendingVisibleUpdates;

llvm::SmallPtrSet<Decl *, 4> PendingDefinitions;

using PendingBodiesMap =
llvm::MapVector<Decl *, uint64_t,
llvm::SmallDenseMap<Decl *, unsigned, 4>,
SmallVector<std::pair<Decl *, uint64_t>, 4>>;

PendingBodiesMap PendingBodies;

llvm::SetVector<NamedDecl *> PendingMergedDefinitionsToDeduplicate;

bool ReadLexicalDeclContextStorage(ModuleFile &M,
llvm::BitstreamCursor &Cursor,
uint64_t Offset, DeclContext *DC);

bool ReadVisibleDeclContextStorage(ModuleFile &M,
llvm::BitstreamCursor &Cursor,
uint64_t Offset, serialization::DeclID ID);

std::vector<IdentifierInfo *> IdentifiersLoaded;

using GlobalIdentifierMapType =
ContinuousRangeMap<serialization::IdentID, ModuleFile *, 4>;

GlobalIdentifierMapType GlobalIdentifierMap;

std::vector<MacroInfo *> MacrosLoaded;

using LoadedMacroInfo =
std::pair<IdentifierInfo *, serialization::SubmoduleID>;

llvm::DenseSet<LoadedMacroInfo> LoadedUndefs;

using GlobalMacroMapType =
ContinuousRangeMap<serialization::MacroID, ModuleFile *, 4>;

GlobalMacroMapType GlobalMacroMap;

SmallVector<Module *, 2> SubmodulesLoaded;

using GlobalSubmoduleMapType =
ContinuousRangeMap<serialization::SubmoduleID, ModuleFile *, 4>;

GlobalSubmoduleMapType GlobalSubmoduleMap;

using HiddenNames = SmallVector<Decl *, 2>;
using HiddenNamesMapType = llvm::DenseMap<Module *, HiddenNames>;

HiddenNamesMapType HiddenNamesMap;

struct UnresolvedModuleRef {
ModuleFile *File;

Module *Mod;

enum { Import, Export, Conflict } Kind;

unsigned ID;

unsigned IsWildcard : 1;

StringRef String;
};

SmallVector<UnresolvedModuleRef, 2> UnresolvedModuleRefs;

SmallVector<Selector, 16> SelectorsLoaded;

using GlobalSelectorMapType =
ContinuousRangeMap<serialization::SelectorID, ModuleFile *, 4>;

GlobalSelectorMapType GlobalSelectorMap;

llvm::DenseMap<Selector, unsigned> SelectorGeneration;

llvm::DenseMap<Selector, bool> SelectorOutOfDate;

struct PendingMacroInfo {
ModuleFile *M;
uint64_t MacroDirectivesOffset;

PendingMacroInfo(ModuleFile *M, uint64_t MacroDirectivesOffset)
: M(M), MacroDirectivesOffset(MacroDirectivesOffset) {}
};

using PendingMacroIDsMap =
llvm::MapVector<IdentifierInfo *, SmallVector<PendingMacroInfo, 2>>;

PendingMacroIDsMap PendingMacroIDs;

using GlobalPreprocessedEntityMapType =
ContinuousRangeMap<unsigned, ModuleFile *, 4>;

GlobalPreprocessedEntityMapType GlobalPreprocessedEntityMap;

using GlobalSkippedRangeMapType =
ContinuousRangeMap<unsigned, ModuleFile *, 4>;

GlobalSkippedRangeMapType GlobalSkippedRangeMap;


SmallVector<uint64_t, 16> EagerlyDeserializedDecls;

SmallVector<uint64_t, 16> TentativeDefinitions;

SmallVector<uint64_t, 64> VTableUses;

SmallVector<uint64_t, 64> PendingInstantiations;



SmallVector<uint64_t, 16> UnusedFileScopedDecls;

SmallVector<uint64_t, 4> DelegatingCtorDecls;

SmallVector<uint64_t, 64> ReferencedSelectorsData;

SmallVector<uint64_t, 64> WeakUndeclaredIdentifiers;

SmallVector<uint64_t, 4> ExtVectorDecls;



SmallVector<uint64_t, 16> UnusedLocalTypedefNameCandidates;

unsigned ForceCUDAHostDeviceDepth = 0;

SmallVector<uint64_t, 4> SemaDeclRefs;

SmallVector<uint64_t, 16> SpecialTypes;

SmallVector<uint64_t, 2> CUDASpecialDeclRefs;

SmallVector<uint64_t, 1> FPPragmaOptions;

SourceLocation OptimizeOffPragmaLocation;

int PragmaMSStructState = -1;

int PragmaMSPointersToMembersState = -1;
SourceLocation PointersToMembersPragmaLocation;

Optional<unsigned> PragmaPackCurrentValue;
SourceLocation PragmaPackCurrentLocation;
struct PragmaPackStackEntry {
unsigned Value;
SourceLocation Location;
SourceLocation PushLocation;
StringRef SlotLabel;
};
llvm::SmallVector<PragmaPackStackEntry, 2> PragmaPackStack;
llvm::SmallVector<std::string, 2> PragmaPackStrings;

OpenCLOptions OpenCLExtensions;

llvm::DenseMap<const Type *, std::set<std::string>> OpenCLTypeExtMap;

llvm::DenseMap<const Decl *, std::set<std::string>> OpenCLDeclExtMap;

SmallVector<uint64_t, 4> KnownNamespaces;

SmallVector<uint64_t, 8> UndefinedButUsed;

SmallVector<uint64_t, 8> DelayedDeleteExprs;

SmallVector<uint64_t, 1> LateParsedTemplates;

public:
struct ImportedSubmodule {
serialization::SubmoduleID ID;
SourceLocation ImportLoc;

ImportedSubmodule(serialization::SubmoduleID ID, SourceLocation ImportLoc)
: ID(ID), ImportLoc(ImportLoc) {}
};

private:
SmallVector<ImportedSubmodule, 2> ImportedModules;

std::string isysroot;

bool DisableValidation;

bool AllowASTWithCompilerErrors;

bool AllowConfigurationMismatch;

bool ValidateSystemInputs;

bool UseGlobalIndex;

bool TriedLoadingGlobalIndex = false;

bool ProcessingUpdateRecords = false;

using SwitchCaseMapTy = llvm::DenseMap<unsigned, SwitchCase *>;

SwitchCaseMapTy SwitchCaseStmts;

SwitchCaseMapTy *CurrSwitchCaseStmts;

unsigned NumSLocEntriesRead = 0;

unsigned TotalNumSLocEntries = 0;

unsigned NumStatementsRead = 0;

unsigned TotalNumStatements = 0;

unsigned NumMacrosRead = 0;

unsigned TotalNumMacros = 0;

unsigned NumIdentifierLookups = 0;

unsigned NumIdentifierLookupHits = 0;

unsigned NumSelectorsRead = 0;

unsigned NumMethodPoolEntriesRead = 0;

unsigned NumMethodPoolLookups = 0;

unsigned NumMethodPoolHits = 0;

unsigned NumMethodPoolTableLookups = 0;

unsigned NumMethodPoolTableHits = 0;

unsigned TotalNumMethodPoolEntries = 0;

unsigned NumLexicalDeclContextsRead = 0, TotalLexicalDeclContexts = 0;

unsigned NumVisibleDeclContextsRead = 0, TotalVisibleDeclContexts = 0;

uint64_t TotalModulesSizeInBits = 0;

unsigned NumCurrentElementsDeserializing = 0;

bool PassingDeclsToConsumer = false;

llvm::MapVector<IdentifierInfo *, SmallVector<uint32_t, 4>>
PendingIdentifierInfos;

llvm::SmallMapVector<IdentifierInfo *, SmallVector<NamedDecl*, 2>, 16>
PendingFakeLookupResults;

llvm::DenseMap<IdentifierInfo *, unsigned> IdentifierGeneration;

class InterestingDecl {
Decl *D;
bool DeclHasPendingBody;

public:
InterestingDecl(Decl *D, bool HasBody)
: D(D), DeclHasPendingBody(HasBody) {}

Decl *getDecl() { return D; }

bool hasPendingBody() { return DeclHasPendingBody; }
};

std::deque<InterestingDecl> PotentiallyInterestingDecls;

SmallVector<std::pair<Decl *, uint64_t>, 16> PendingDeclChains;

SmallVector<Decl *, 16> PendingIncompleteDeclChains;

struct PendingDeclContextInfo {
Decl *D;
serialization::GlobalDeclID SemaDC;
serialization::GlobalDeclID LexicalDC;
};

std::deque<PendingDeclContextInfo> PendingDeclContextInfos;

llvm::SmallVector<NamedDecl *, 16> PendingOdrMergeChecks;

using DataPointers =
std::pair<CXXRecordDecl *, struct CXXRecordDecl::DefinitionData *>;

llvm::SmallDenseMap<CXXRecordDecl *, llvm::SmallVector<DataPointers, 2>, 2>
PendingOdrMergeFailures;

llvm::SmallDenseMap<FunctionDecl *, llvm::SmallVector<FunctionDecl *, 2>, 2>
PendingFunctionOdrMergeFailures;

llvm::SmallDenseMap<EnumDecl *, llvm::SmallVector<EnumDecl *, 2>, 2>
PendingEnumOdrMergeFailures;

llvm::SmallPtrSet<DeclContext*, 2> DiagnosedOdrMergeFailures;

llvm::SmallPtrSet<ObjCCategoryDecl *, 16> CategoriesDeserialized;

SmallVector<ObjCInterfaceDecl *, 16> ObjCClassesLoaded;

using KeyDeclsMap =
llvm::DenseMap<Decl *, SmallVector<serialization::DeclID, 2>>;

KeyDeclsMap KeyDecls;

llvm::DenseMap<DeclContext *, DeclContext *> MergedDeclContexts;

llvm::DenseMap<EnumDecl *, EnumDecl *> EnumDefinitions;

SmallVector<Stmt *, 16> StmtStack;

enum ReadingKind {
Read_None, Read_Decl, Read_Type, Read_Stmt
};

ReadingKind ReadingKind = Read_None;

class ReadingKindTracker {
ASTReader &Reader;
enum ReadingKind PrevKind;

public:
ReadingKindTracker(enum ReadingKind newKind, ASTReader &reader)
: Reader(reader), PrevKind(Reader.ReadingKind) {
Reader.ReadingKind = newKind;
}

ReadingKindTracker(const ReadingKindTracker &) = delete;
ReadingKindTracker &operator=(const ReadingKindTracker &) = delete;
~ReadingKindTracker() { Reader.ReadingKind = PrevKind; }
};

class ProcessingUpdatesRAIIObj {
ASTReader &Reader;
bool PrevState;

public:
ProcessingUpdatesRAIIObj(ASTReader &reader)
: Reader(reader), PrevState(Reader.ProcessingUpdateRecords) {
Reader.ProcessingUpdateRecords = true;
}

ProcessingUpdatesRAIIObj(const ProcessingUpdatesRAIIObj &) = delete;
ProcessingUpdatesRAIIObj &
operator=(const ProcessingUpdatesRAIIObj &) = delete;
~ProcessingUpdatesRAIIObj() { Reader.ProcessingUpdateRecords = PrevState; }
};

std::string SuggestedPredefines;

llvm::DenseMap<const Decl *, bool> DefinitionSource;

Stmt *ReadStmtFromStream(ModuleFile &F);

struct InputFileInfo {
std::string Filename;
off_t StoredSize;
time_t StoredTime;
bool Overridden;
bool Transient;
bool TopLevelModuleMap;
};

InputFileInfo readInputFileInfo(ModuleFile &F, unsigned ID);

serialization::InputFile getInputFile(ModuleFile &F, unsigned ID,
bool Complain = true);

public:
void ResolveImportedPath(ModuleFile &M, std::string &Filename);
static void ResolveImportedPath(std::string &Filename, StringRef Prefix);

Decl *getKeyDeclaration(Decl *D) {
D = D->getCanonicalDecl();
if (D->isFromASTFile())
return D;

auto I = KeyDecls.find(D);
if (I == KeyDecls.end() || I->second.empty())
return D;
return GetExistingDecl(I->second[0]);
}
const Decl *getKeyDeclaration(const Decl *D) {
return getKeyDeclaration(const_cast<Decl*>(D));
}

template <typename Fn>
void forEachImportedKeyDecl(const Decl *D, Fn Visit) {
D = D->getCanonicalDecl();
if (D->isFromASTFile())
Visit(D);

auto It = KeyDecls.find(const_cast<Decl*>(D));
if (It != KeyDecls.end())
for (auto ID : It->second)
Visit(GetExistingDecl(ID));
}

const serialization::reader::DeclContextLookupTable *
getLoadedLookupTables(DeclContext *Primary) const;

private:
struct ImportedModule {
ModuleFile *Mod;
ModuleFile *ImportedBy;
SourceLocation ImportLoc;

ImportedModule(ModuleFile *Mod,
ModuleFile *ImportedBy,
SourceLocation ImportLoc)
: Mod(Mod), ImportedBy(ImportedBy), ImportLoc(ImportLoc) {}
};

ASTReadResult ReadASTCore(StringRef FileName, ModuleKind Type,
SourceLocation ImportLoc, ModuleFile *ImportedBy,
SmallVectorImpl<ImportedModule> &Loaded,
off_t ExpectedSize, time_t ExpectedModTime,
ASTFileSignature ExpectedSignature,
unsigned ClientLoadCapabilities);
ASTReadResult ReadControlBlock(ModuleFile &F,
SmallVectorImpl<ImportedModule> &Loaded,
const ModuleFile *ImportedBy,
unsigned ClientLoadCapabilities);
static ASTReadResult ReadOptionsBlock(
llvm::BitstreamCursor &Stream, unsigned ClientLoadCapabilities,
bool AllowCompatibleConfigurationMismatch, ASTReaderListener &Listener,
std::string &SuggestedPredefines);

ASTReadResult readUnhashedControlBlock(ModuleFile &F, bool WasImportedBy,
unsigned ClientLoadCapabilities);

static ASTReadResult
readUnhashedControlBlockImpl(ModuleFile *F, llvm::StringRef StreamData,
unsigned ClientLoadCapabilities,
bool AllowCompatibleConfigurationMismatch,
ASTReaderListener *Listener,
bool ValidateDiagnosticOptions);

ASTReadResult ReadASTBlock(ModuleFile &F, unsigned ClientLoadCapabilities);
ASTReadResult ReadExtensionBlock(ModuleFile &F);
void ReadModuleOffsetMap(ModuleFile &F) const;
bool ParseLineTable(ModuleFile &F, const RecordData &Record);
bool ReadSourceManagerBlock(ModuleFile &F);
llvm::BitstreamCursor &SLocCursorForID(int ID);
SourceLocation getImportLocation(ModuleFile *F);
ASTReadResult ReadModuleMapFileBlock(RecordData &Record, ModuleFile &F,
const ModuleFile *ImportedBy,
unsigned ClientLoadCapabilities);
ASTReadResult ReadSubmoduleBlock(ModuleFile &F,
unsigned ClientLoadCapabilities);
static bool ParseLanguageOptions(const RecordData &Record, bool Complain,
ASTReaderListener &Listener,
bool AllowCompatibleDifferences);
static bool ParseTargetOptions(const RecordData &Record, bool Complain,
ASTReaderListener &Listener,
bool AllowCompatibleDifferences);
static bool ParseDiagnosticOptions(const RecordData &Record, bool Complain,
ASTReaderListener &Listener);
static bool ParseFileSystemOptions(const RecordData &Record, bool Complain,
ASTReaderListener &Listener);
static bool ParseHeaderSearchOptions(const RecordData &Record, bool Complain,
ASTReaderListener &Listener);
static bool ParsePreprocessorOptions(const RecordData &Record, bool Complain,
ASTReaderListener &Listener,
std::string &SuggestedPredefines);

struct RecordLocation {
ModuleFile *F;
uint64_t Offset;

RecordLocation(ModuleFile *M, uint64_t O) : F(M), Offset(O) {}
};

QualType readTypeRecord(unsigned Index);
void readExceptionSpec(ModuleFile &ModuleFile,
SmallVectorImpl<QualType> &ExceptionStorage,
FunctionProtoType::ExceptionSpecInfo &ESI,
const RecordData &Record, unsigned &Index);
RecordLocation TypeCursorForIndex(unsigned Index);
void LoadedDecl(unsigned Index, Decl *D);
Decl *ReadDeclRecord(serialization::DeclID ID);
void markIncompleteDeclChain(Decl *Canon);

Decl *getMostRecentExistingDecl(Decl *D);

RecordLocation DeclCursorForID(serialization::DeclID ID,
SourceLocation &Location);
void loadDeclUpdateRecords(PendingUpdateRecord &Record);
void loadPendingDeclChain(Decl *D, uint64_t LocalOffset);
void loadObjCCategories(serialization::GlobalDeclID ID, ObjCInterfaceDecl *D,
unsigned PreviousGeneration = 0);

RecordLocation getLocalBitOffset(uint64_t GlobalOffset);
uint64_t getGlobalBitOffset(ModuleFile &M, uint32_t LocalOffset);

serialization::PreprocessedEntityID
findPreprocessedEntity(SourceLocation Loc, bool EndsAfter) const;

serialization::PreprocessedEntityID
findNextPreprocessedEntity(
GlobalSLocOffsetMapType::const_iterator SLocMapI) const;

std::pair<ModuleFile *, unsigned>
getModulePreprocessedEntity(unsigned GlobalIndex);

llvm::iterator_range<PreprocessingRecord::iterator>
getModulePreprocessedEntities(ModuleFile &Mod) const;

public:
class ModuleDeclIterator
: public llvm::iterator_adaptor_base<
ModuleDeclIterator, const serialization::LocalDeclID *,
std::random_access_iterator_tag, const Decl *, ptrdiff_t,
const Decl *, const Decl *> {
ASTReader *Reader = nullptr;
ModuleFile *Mod = nullptr;

public:
ModuleDeclIterator() : iterator_adaptor_base(nullptr) {}

ModuleDeclIterator(ASTReader *Reader, ModuleFile *Mod,
const serialization::LocalDeclID *Pos)
: iterator_adaptor_base(Pos), Reader(Reader), Mod(Mod) {}

value_type operator*() const {
return Reader->GetDecl(Reader->getGlobalDeclID(*Mod, *I));
}

value_type operator->() const { return **this; }

bool operator==(const ModuleDeclIterator &RHS) const {
assert(Reader == RHS.Reader && Mod == RHS.Mod);
return I == RHS.I;
}
};

llvm::iterator_range<ModuleDeclIterator>
getModuleFileLevelDecls(ModuleFile &Mod);

private:
void PassInterestingDeclsToConsumer();
void PassInterestingDeclToConsumer(Decl *D);

void finishPendingActions();
void diagnoseOdrViolations();

void pushExternalDeclIntoScope(NamedDecl *D, DeclarationName Name);

void addPendingDeclContextInfo(Decl *D,
serialization::GlobalDeclID SemaDC,
serialization::GlobalDeclID LexicalDC) {
assert(D);
PendingDeclContextInfo Info = { D, SemaDC, LexicalDC };
PendingDeclContextInfos.push_back(Info);
}

void Error(StringRef Msg) const;
void Error(unsigned DiagID, StringRef Arg1 = StringRef(),
StringRef Arg2 = StringRef()) const;

public:
ASTReader(Preprocessor &PP, ASTContext *Context,
const PCHContainerReader &PCHContainerRdr,
ArrayRef<std::shared_ptr<ModuleFileExtension>> Extensions,
StringRef isysroot = "", bool DisableValidation = false,
bool AllowASTWithCompilerErrors = false,
bool AllowConfigurationMismatch = false,
bool ValidateSystemInputs = false, bool UseGlobalIndex = true,
std::unique_ptr<llvm::Timer> ReadTimer = {});
ASTReader(const ASTReader &) = delete;
ASTReader &operator=(const ASTReader &) = delete;
~ASTReader() override;

SourceManager &getSourceManager() const { return SourceMgr; }
FileManager &getFileManager() const { return FileMgr; }
DiagnosticsEngine &getDiags() const { return Diags; }

enum LoadFailureCapabilities {
ARR_None = 0,

ARR_Missing = 0x1,

ARR_OutOfDate = 0x2,

ARR_VersionMismatch = 0x4,

ARR_ConfigurationMismatch = 0x8
};

ASTReadResult ReadAST(StringRef FileName, ModuleKind Type,
SourceLocation ImportLoc,
unsigned ClientLoadCapabilities,
SmallVectorImpl<ImportedSubmodule> *Imported = nullptr);

void makeModuleVisible(Module *Mod,
Module::NameVisibilityKind NameVisibility,
SourceLocation ImportLoc);

void makeNamesVisible(const HiddenNames &Names, Module *Owner);

void mergeDefinitionVisibility(NamedDecl *Def, NamedDecl *MergedDef);

std::unique_ptr<ASTReaderListener> takeListener() {
return std::move(Listener);
}

void setListener(std::unique_ptr<ASTReaderListener> Listener) {
this->Listener = std::move(Listener);
}

void addListener(std::unique_ptr<ASTReaderListener> L) {
if (Listener)
L = llvm::make_unique<ChainedASTReaderListener>(std::move(L),
std::move(Listener));
Listener = std::move(L);
}

class ListenerScope {
ASTReader &Reader;
bool Chained = false;

public:
ListenerScope(ASTReader &Reader, std::unique_ptr<ASTReaderListener> L)
: Reader(Reader) {
auto Old = Reader.takeListener();
if (Old) {
Chained = true;
L = llvm::make_unique<ChainedASTReaderListener>(std::move(L),
std::move(Old));
}
Reader.setListener(std::move(L));
}

~ListenerScope() {
auto New = Reader.takeListener();
if (Chained)
Reader.setListener(static_cast<ChainedASTReaderListener *>(New.get())
->takeSecond());
}
};

void setDeserializationListener(ASTDeserializationListener *Listener,
bool TakeOwnership = false);

ASTDeserializationListener *getDeserializationListener() {
return DeserializationListener;
}

bool hasGlobalIndex() const { return (bool)GlobalIndex; }

GlobalModuleIndex *getGlobalIndex() { return GlobalIndex.get(); }

void resetForReload() { TriedLoadingGlobalIndex = false; }

bool loadGlobalIndex();

bool isGlobalIndexUnavailable() const;

void InitializeContext();

void UpdateSema();

void addInMemoryBuffer(StringRef &FileName,
std::unique_ptr<llvm::MemoryBuffer> Buffer) {
ModuleMgr.addInMemoryBuffer(FileName, std::move(Buffer));
}

void finalizeForWriting();

ModuleManager &getModuleManager() { return ModuleMgr; }

Preprocessor &getPreprocessor() const { return PP; }

StringRef getOriginalSourceFile() {
return ModuleMgr.getPrimaryModule().OriginalSourceFileName;
}

static std::string
getOriginalSourceFile(const std::string &ASTFileName, FileManager &FileMgr,
const PCHContainerReader &PCHContainerRdr,
DiagnosticsEngine &Diags);

static bool
readASTFileControlBlock(StringRef Filename, FileManager &FileMgr,
const PCHContainerReader &PCHContainerRdr,
bool FindModuleFileExtensions,
ASTReaderListener &Listener,
bool ValidateDiagnosticOptions);

static bool isAcceptableASTFile(StringRef Filename, FileManager &FileMgr,
const PCHContainerReader &PCHContainerRdr,
const LangOptions &LangOpts,
const TargetOptions &TargetOpts,
const PreprocessorOptions &PPOpts,
StringRef ExistingModuleCachePath);

const std::string &getSuggestedPredefines() { return SuggestedPredefines; }

PreprocessedEntity *ReadPreprocessedEntity(unsigned Index) override;

std::pair<unsigned, unsigned>
findPreprocessedEntitiesInRange(SourceRange Range) override;

Optional<bool> isPreprocessedEntityInFileID(unsigned Index,
FileID FID) override;

SourceRange ReadSkippedRange(unsigned Index) override;

HeaderFileInfo GetHeaderFileInfo(const FileEntry *FE) override;

void ReadPragmaDiagnosticMappings(DiagnosticsEngine &Diag);

unsigned getTotalNumSLocs() const {
return TotalNumSLocEntries;
}

unsigned getTotalNumIdentifiers() const {
return static_cast<unsigned>(IdentifiersLoaded.size());
}

unsigned getTotalNumMacros() const {
return static_cast<unsigned>(MacrosLoaded.size());
}

unsigned getTotalNumTypes() const {
return static_cast<unsigned>(TypesLoaded.size());
}

unsigned getTotalNumDecls() const {
return static_cast<unsigned>(DeclsLoaded.size());
}

unsigned getTotalNumSubmodules() const {
return static_cast<unsigned>(SubmodulesLoaded.size());
}

unsigned getTotalNumSelectors() const {
return static_cast<unsigned>(SelectorsLoaded.size());
}

unsigned getTotalNumPreprocessedEntities() const {
unsigned Result = 0;
for (const auto &M : ModuleMgr)
Result += M.NumPreprocessedEntities;
return Result;
}

TemplateArgumentLocInfo
GetTemplateArgumentLocInfo(ModuleFile &F, TemplateArgument::ArgKind Kind,
const RecordData &Record, unsigned &Idx);

TemplateArgumentLoc
ReadTemplateArgumentLoc(ModuleFile &F,
const RecordData &Record, unsigned &Idx);

const ASTTemplateArgumentListInfo*
ReadASTTemplateArgumentListInfo(ModuleFile &F,
const RecordData &Record, unsigned &Index);

TypeSourceInfo *GetTypeSourceInfo(ModuleFile &F,
const RecordData &Record, unsigned &Idx);

void ReadTypeLoc(ModuleFile &F, const RecordData &Record, unsigned &Idx,
TypeLoc TL);

QualType GetType(serialization::TypeID ID);

QualType getLocalType(ModuleFile &F, unsigned LocalID);

serialization::TypeID getGlobalTypeID(ModuleFile &F, unsigned LocalID) const;

QualType readType(ModuleFile &F, const RecordData &Record, unsigned &Idx) {
if (Idx >= Record.size())
return {};

return getLocalType(F, Record[Idx++]);
}

serialization::DeclID getGlobalDeclID(ModuleFile &F,
serialization::LocalDeclID LocalID) const;

bool isDeclIDFromModule(serialization::GlobalDeclID ID, ModuleFile &M) const;

ModuleFile *getOwningModuleFile(const Decl *D);

std::string getOwningModuleNameForDiagnostic(const Decl *D);

SourceLocation getSourceLocationForDeclID(serialization::GlobalDeclID ID);

Decl *GetDecl(serialization::DeclID ID);
Decl *GetExternalDecl(uint32_t ID) override;

Decl *GetExistingDecl(serialization::DeclID ID);

Decl *GetLocalDecl(ModuleFile &F, uint32_t LocalID) {
return GetDecl(getGlobalDeclID(F, LocalID));
}

template<typename T>
T *GetLocalDeclAs(ModuleFile &F, uint32_t LocalID) {
return cast_or_null<T>(GetLocalDecl(F, LocalID));
}

serialization::DeclID
mapGlobalIDToModuleFileGlobalID(ModuleFile &M,
serialization::DeclID GlobalID);

serialization::DeclID ReadDeclID(ModuleFile &F, const RecordData &Record,
unsigned &Idx);

Decl *ReadDecl(ModuleFile &F, const RecordData &R, unsigned &I) {
return GetDecl(ReadDeclID(F, R, I));
}

template<typename T>
T *ReadDeclAs(ModuleFile &F, const RecordData &R, unsigned &I) {
return cast_or_null<T>(GetDecl(ReadDeclID(F, R, I)));
}

void CompleteRedeclChain(const Decl *D) override;

CXXBaseSpecifier *GetExternalCXXBaseSpecifiers(uint64_t Offset) override;

Stmt *GetExternalDeclStmt(uint64_t Offset) override;

static bool ReadBlockAbbrevs(llvm::BitstreamCursor &Cursor, unsigned BlockID);

bool FindExternalVisibleDeclsByName(const DeclContext *DC,
DeclarationName Name) override;

void
FindExternalLexicalDecls(const DeclContext *DC,
llvm::function_ref<bool(Decl::Kind)> IsKindWeWant,
SmallVectorImpl<Decl *> &Decls) override;

void FindFileRegionDecls(FileID File, unsigned Offset, unsigned Length,
SmallVectorImpl<Decl *> &Decls) override;

void StartedDeserializing() override;

void FinishedDeserializing() override;

void StartTranslationUnit(ASTConsumer *Consumer) override;

void PrintStats() override;

void dump();

void getMemoryBufferSizes(MemoryBufferSizes &sizes) const override;

void InitializeSema(Sema &S) override;

void ForgetSema() override { SemaObj = nullptr; }

IdentifierInfo *get(StringRef Name) override;

IdentifierIterator *getIdentifiers() override;

void ReadMethodPool(Selector Sel) override;

void updateOutOfDateSelector(Selector Sel) override;

void ReadKnownNamespaces(
SmallVectorImpl<NamespaceDecl *> &Namespaces) override;

void ReadUndefinedButUsed(
llvm::MapVector<NamedDecl *, SourceLocation> &Undefined) override;

void ReadMismatchingDeleteExpressions(llvm::MapVector<
FieldDecl *, llvm::SmallVector<std::pair<SourceLocation, bool>, 4>> &
Exprs) override;

void ReadTentativeDefinitions(
SmallVectorImpl<VarDecl *> &TentativeDefs) override;

void ReadUnusedFileScopedDecls(
SmallVectorImpl<const DeclaratorDecl *> &Decls) override;

void ReadDelegatingConstructors(
SmallVectorImpl<CXXConstructorDecl *> &Decls) override;

void ReadExtVectorDecls(SmallVectorImpl<TypedefNameDecl *> &Decls) override;

void ReadUnusedLocalTypedefNameCandidates(
llvm::SmallSetVector<const TypedefNameDecl *, 4> &Decls) override;

void ReadReferencedSelectors(
SmallVectorImpl<std::pair<Selector, SourceLocation>> &Sels) override;

void ReadWeakUndeclaredIdentifiers(
SmallVectorImpl<std::pair<IdentifierInfo *, WeakInfo>> &WI) override;

void ReadUsedVTables(SmallVectorImpl<ExternalVTableUse> &VTables) override;

void ReadPendingInstantiations(
SmallVectorImpl<std::pair<ValueDecl *,
SourceLocation>> &Pending) override;

void ReadLateParsedTemplates(
llvm::MapVector<const FunctionDecl *, std::unique_ptr<LateParsedTemplate>>
&LPTMap) override;

void LoadSelector(Selector Sel);

void SetIdentifierInfo(unsigned ID, IdentifierInfo *II);
void SetGloballyVisibleDecls(IdentifierInfo *II,
const SmallVectorImpl<uint32_t> &DeclIDs,
SmallVectorImpl<Decl *> *Decls = nullptr);

DiagnosticBuilder Diag(unsigned DiagID) const;

DiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID) const;

IdentifierInfo *DecodeIdentifierInfo(serialization::IdentifierID ID);

IdentifierInfo *GetIdentifierInfo(ModuleFile &M, const RecordData &Record,
unsigned &Idx) {
return DecodeIdentifierInfo(getGlobalIdentifierID(M, Record[Idx++]));
}

IdentifierInfo *GetIdentifier(serialization::IdentifierID ID) override {
Deserializing AnIdentifier(this);

return DecodeIdentifierInfo(ID);
}

IdentifierInfo *getLocalIdentifier(ModuleFile &M, unsigned LocalID);

serialization::IdentifierID getGlobalIdentifierID(ModuleFile &M,
unsigned LocalID);

void resolvePendingMacro(IdentifierInfo *II, const PendingMacroInfo &PMInfo);

MacroInfo *getMacro(serialization::MacroID ID);

serialization::MacroID getGlobalMacroID(ModuleFile &M, unsigned LocalID);

bool ReadSLocEntry(int ID) override;

std::pair<SourceLocation, StringRef> getModuleImportLoc(int ID) override;

serialization::SubmoduleID
getGlobalSubmoduleID(ModuleFile &M, unsigned LocalID);

Module *getSubmodule(serialization::SubmoduleID GlobalID);

Module *getModule(unsigned ID) override;

bool DeclIsFromPCHWithObjectFile(const Decl *D) override;

ModuleFile *getLocalModuleFile(ModuleFile &M, unsigned ID);

unsigned getModuleFileID(ModuleFile *M);

llvm::Optional<ASTSourceDescriptor> getSourceDescriptor(unsigned ID) override;

ExtKind hasExternalDefinitions(const Decl *D) override;

Selector getLocalSelector(ModuleFile &M, unsigned LocalID);

Selector DecodeSelector(serialization::SelectorID Idx);

Selector GetExternalSelector(serialization::SelectorID ID) override;
uint32_t GetNumExternalSelectors() override;

Selector ReadSelector(ModuleFile &M, const RecordData &Record, unsigned &Idx) {
return getLocalSelector(M, Record[Idx++]);
}

serialization::SelectorID getGlobalSelectorID(ModuleFile &F,
unsigned LocalID) const;

DeclarationName ReadDeclarationName(ModuleFile &F,
const RecordData &Record, unsigned &Idx);
void ReadDeclarationNameLoc(ModuleFile &F,
DeclarationNameLoc &DNLoc, DeclarationName Name,
const RecordData &Record, unsigned &Idx);
void ReadDeclarationNameInfo(ModuleFile &F, DeclarationNameInfo &NameInfo,
const RecordData &Record, unsigned &Idx);

void ReadQualifierInfo(ModuleFile &F, QualifierInfo &Info,
const RecordData &Record, unsigned &Idx);

NestedNameSpecifier *ReadNestedNameSpecifier(ModuleFile &F,
const RecordData &Record,
unsigned &Idx);

NestedNameSpecifierLoc ReadNestedNameSpecifierLoc(ModuleFile &F,
const RecordData &Record,
unsigned &Idx);

TemplateName ReadTemplateName(ModuleFile &F, const RecordData &Record,
unsigned &Idx);

TemplateArgument ReadTemplateArgument(ModuleFile &F, const RecordData &Record,
unsigned &Idx,
bool Canonicalize = false);

TemplateParameterList *ReadTemplateParameterList(ModuleFile &F,
const RecordData &Record,
unsigned &Idx);

void ReadTemplateArgumentList(SmallVectorImpl<TemplateArgument> &TemplArgs,
ModuleFile &F, const RecordData &Record,
unsigned &Idx, bool Canonicalize = false);

void ReadUnresolvedSet(ModuleFile &F, LazyASTUnresolvedSet &Set,
const RecordData &Record, unsigned &Idx);

CXXBaseSpecifier ReadCXXBaseSpecifier(ModuleFile &F,
const RecordData &Record,unsigned &Idx);

CXXCtorInitializer **
ReadCXXCtorInitializers(ModuleFile &F, const RecordData &Record,
unsigned &Idx);

CXXCtorInitializer **GetExternalCXXCtorInitializers(uint64_t Offset) override;

SourceLocation ReadUntranslatedSourceLocation(uint32_t Raw) const {
return SourceLocation::getFromRawEncoding((Raw >> 1) | (Raw << 31));
}

SourceLocation ReadSourceLocation(ModuleFile &ModuleFile, uint32_t Raw) const {
SourceLocation Loc = ReadUntranslatedSourceLocation(Raw);
return TranslateSourceLocation(ModuleFile, Loc);
}

SourceLocation TranslateSourceLocation(ModuleFile &ModuleFile,
SourceLocation Loc) const {
if (!ModuleFile.ModuleOffsetMap.empty())
ReadModuleOffsetMap(ModuleFile);
assert(ModuleFile.SLocRemap.find(Loc.getOffset()) !=
ModuleFile.SLocRemap.end() &&
"Cannot find offset to remap.");
int Remap = ModuleFile.SLocRemap.find(Loc.getOffset())->second;
return Loc.getLocWithOffset(Remap);
}

SourceLocation ReadSourceLocation(ModuleFile &ModuleFile,
const RecordDataImpl &Record,
unsigned &Idx) {
return ReadSourceLocation(ModuleFile, Record[Idx++]);
}

SourceRange ReadSourceRange(ModuleFile &F,
const RecordData &Record, unsigned &Idx);

llvm::APInt ReadAPInt(const RecordData &Record, unsigned &Idx);

llvm::APSInt ReadAPSInt(const RecordData &Record, unsigned &Idx);

llvm::APFloat ReadAPFloat(const RecordData &Record,
const llvm::fltSemantics &Sem, unsigned &Idx);

static std::string ReadString(const RecordData &Record, unsigned &Idx);

static void SkipString(const RecordData &Record, unsigned &Idx) {
Idx += Record[Idx] + 1;
}

std::string ReadPath(ModuleFile &F, const RecordData &Record, unsigned &Idx);

static void SkipPath(const RecordData &Record, unsigned &Idx) {
SkipString(Record, Idx);
}

static VersionTuple ReadVersionTuple(const RecordData &Record, unsigned &Idx);

CXXTemporary *ReadCXXTemporary(ModuleFile &F, const RecordData &Record,
unsigned &Idx);

void ReadAttributes(ASTRecordReader &Record, AttrVec &Attrs);

Stmt *ReadStmt(ModuleFile &F);

Expr *ReadExpr(ModuleFile &F);

Stmt *ReadSubStmt() {
assert(ReadingKind == Read_Stmt &&
"Should be called only during statement reading!");
assert(!StmtStack.empty() && "Read too many sub-statements!");
return StmtStack.pop_back_val();
}

Expr *ReadSubExpr();

Token ReadToken(ModuleFile &M, const RecordDataImpl &Record, unsigned &Idx);

MacroInfo *ReadMacroRecord(ModuleFile &F, uint64_t Offset);

serialization::PreprocessedEntityID
getGlobalPreprocessedEntityID(ModuleFile &M, unsigned LocalID) const;

void addPendingMacro(IdentifierInfo *II, ModuleFile *M,
uint64_t MacroDirectivesOffset);

void ReadDefinedMacros() override;

void updateOutOfDateIdentifier(IdentifierInfo &II) override;

void markIdentifierUpToDate(IdentifierInfo *II);

void completeVisibleDeclsMap(const DeclContext *DC) override;

ASTContext &getContext() {
assert(ContextObj && "requested AST context when not loading AST");
return *ContextObj;
}

SmallVector<uint64_t, 16> PreloadedDeclIDs;

Sema *getSema() { return SemaObj; }

IdentifierResolver &getIdResolver();

IdentifierTable &getIdentifierTable();

void RecordSwitchCaseID(SwitchCase *SC, unsigned ID);

SwitchCase *getSwitchCaseWithID(unsigned ID);

void ClearSwitchCaseIDs();

SmallVector<std::pair<llvm::BitstreamCursor,
serialization::ModuleFile *>, 8> CommentsCursors;

void ReadComments() override;

void visitInputFiles(serialization::ModuleFile &MF,
bool IncludeSystem, bool Complain,
llvm::function_ref<void(const serialization::InputFile &IF,
bool isSystem)> Visitor);

void visitTopLevelModuleMaps(serialization::ModuleFile &MF,
llvm::function_ref<
void(const FileEntry *)> Visitor);

bool isProcessingUpdateRecords() { return ProcessingUpdateRecords; }
};

class ASTRecordReader {
using ModuleFile = serialization::ModuleFile;

ASTReader *Reader;
ModuleFile *F;
unsigned Idx = 0;
ASTReader::RecordData Record;

using RecordData = ASTReader::RecordData;
using RecordDataImpl = ASTReader::RecordDataImpl;

public:
ASTRecordReader(ASTReader &Reader, ModuleFile &F) : Reader(&Reader), F(&F) {}

unsigned readRecord(llvm::BitstreamCursor &Cursor, unsigned AbbrevID);

bool isModule() const { return F->isModule(); }

ASTContext &getContext() { return Reader->getContext(); }

unsigned getIdx() const { return Idx; }

size_t size() const { return Record.size(); }

const uint64_t &operator[](size_t N) { return Record[N]; }

const uint64_t &back() const { return Record.back(); }

const uint64_t &readInt() { return Record[Idx++]; }

const uint64_t &peekInt() { return Record[Idx]; }

void skipInts(unsigned N) { Idx += N; }

serialization::SubmoduleID
getGlobalSubmoduleID(unsigned LocalID) {
return Reader->getGlobalSubmoduleID(*F, LocalID);
}

Module *getSubmodule(serialization::SubmoduleID GlobalID) {
return Reader->getSubmodule(GlobalID);
}

bool readLexicalDeclContextStorage(uint64_t Offset, DeclContext *DC) {
return Reader->ReadLexicalDeclContextStorage(*F, F->DeclsCursor, Offset,
DC);
}

bool readVisibleDeclContextStorage(uint64_t Offset,
serialization::DeclID ID) {
return Reader->ReadVisibleDeclContextStorage(*F, F->DeclsCursor, Offset,
ID);
}

void readExceptionSpec(SmallVectorImpl<QualType> &ExceptionStorage,
FunctionProtoType::ExceptionSpecInfo &ESI) {
return Reader->readExceptionSpec(*F, ExceptionStorage, ESI, Record, Idx);
}

uint64_t getGlobalBitOffset(uint32_t LocalOffset) {
return Reader->getGlobalBitOffset(*F, LocalOffset);
}

Stmt *readStmt() { return Reader->ReadStmt(*F); }

Expr *readExpr() { return Reader->ReadExpr(*F); }

Stmt *readSubStmt() { return Reader->ReadSubStmt(); }

Expr *readSubExpr() { return Reader->ReadSubExpr(); }

template<typename T>
T *GetLocalDeclAs(uint32_t LocalID) {
return cast_or_null<T>(Reader->GetLocalDecl(*F, LocalID));
}

TemplateArgumentLocInfo
getTemplateArgumentLocInfo(TemplateArgument::ArgKind Kind) {
return Reader->GetTemplateArgumentLocInfo(*F, Kind, Record, Idx);
}

TemplateArgumentLoc
readTemplateArgumentLoc() {
return Reader->ReadTemplateArgumentLoc(*F, Record, Idx);
}

const ASTTemplateArgumentListInfo*
readASTTemplateArgumentListInfo() {
return Reader->ReadASTTemplateArgumentListInfo(*F, Record, Idx);
}

TypeSourceInfo *getTypeSourceInfo() {
return Reader->GetTypeSourceInfo(*F, Record, Idx);
}

void readTypeLoc(TypeLoc TL) {
return Reader->ReadTypeLoc(*F, Record, Idx, TL);
}

serialization::TypeID getGlobalTypeID(unsigned LocalID) const {
return Reader->getGlobalTypeID(*F, LocalID);
}

QualType readType() {
return Reader->readType(*F, Record, Idx);
}

serialization::DeclID readDeclID() {
return Reader->ReadDeclID(*F, Record, Idx);
}

Decl *readDecl() {
return Reader->ReadDecl(*F, Record, Idx);
}

template<typename T>
T *readDeclAs() {
return Reader->ReadDeclAs<T>(*F, Record, Idx);
}

IdentifierInfo *getIdentifierInfo() {
return Reader->GetIdentifierInfo(*F, Record, Idx);
}

Selector readSelector() {
return Reader->ReadSelector(*F, Record, Idx);
}

DeclarationName readDeclarationName() {
return Reader->ReadDeclarationName(*F, Record, Idx);
}
void readDeclarationNameLoc(DeclarationNameLoc &DNLoc, DeclarationName Name) {
return Reader->ReadDeclarationNameLoc(*F, DNLoc, Name, Record, Idx);
}
void readDeclarationNameInfo(DeclarationNameInfo &NameInfo) {
return Reader->ReadDeclarationNameInfo(*F, NameInfo, Record, Idx);
}

void readQualifierInfo(QualifierInfo &Info) {
return Reader->ReadQualifierInfo(*F, Info, Record, Idx);
}

NestedNameSpecifier *readNestedNameSpecifier() {
return Reader->ReadNestedNameSpecifier(*F, Record, Idx);
}

NestedNameSpecifierLoc readNestedNameSpecifierLoc() {
return Reader->ReadNestedNameSpecifierLoc(*F, Record, Idx);
}

TemplateName readTemplateName() {
return Reader->ReadTemplateName(*F, Record, Idx);
}

TemplateArgument readTemplateArgument(bool Canonicalize = false) {
return Reader->ReadTemplateArgument(*F, Record, Idx, Canonicalize);
}

TemplateParameterList *readTemplateParameterList() {
return Reader->ReadTemplateParameterList(*F, Record, Idx);
}

void readTemplateArgumentList(SmallVectorImpl<TemplateArgument> &TemplArgs,
bool Canonicalize = false) {
return Reader->ReadTemplateArgumentList(TemplArgs, *F, Record, Idx,
Canonicalize);
}

void readUnresolvedSet(LazyASTUnresolvedSet &Set) {
return Reader->ReadUnresolvedSet(*F, Set, Record, Idx);
}

CXXBaseSpecifier readCXXBaseSpecifier() {
return Reader->ReadCXXBaseSpecifier(*F, Record, Idx);
}

CXXCtorInitializer **readCXXCtorInitializers() {
return Reader->ReadCXXCtorInitializers(*F, Record, Idx);
}

CXXTemporary *readCXXTemporary() {
return Reader->ReadCXXTemporary(*F, Record, Idx);
}

SourceLocation readSourceLocation() {
return Reader->ReadSourceLocation(*F, Record, Idx);
}

SourceRange readSourceRange() {
return Reader->ReadSourceRange(*F, Record, Idx);
}

llvm::APInt readAPInt() {
return Reader->ReadAPInt(Record, Idx);
}

llvm::APSInt readAPSInt() {
return Reader->ReadAPSInt(Record, Idx);
}

llvm::APFloat readAPFloat(const llvm::fltSemantics &Sem) {
return Reader->ReadAPFloat(Record, Sem,Idx);
}

std::string readString() {
return Reader->ReadString(Record, Idx);
}

std::string readPath() {
return Reader->ReadPath(*F, Record, Idx);
}

VersionTuple readVersionTuple() {
return ASTReader::ReadVersionTuple(Record, Idx);
}

void readAttributes(AttrVec &Attrs) {
return Reader->ReadAttributes(*this, Attrs);
}

Token readToken() {
return Reader->ReadToken(*F, Record, Idx);
}

void recordSwitchCaseID(SwitchCase *SC, unsigned ID) {
Reader->RecordSwitchCaseID(SC, ID);
}

SwitchCase *getSwitchCaseWithID(unsigned ID) {
return Reader->getSwitchCaseWithID(ID);
}
};

struct SavedStreamPosition {
explicit SavedStreamPosition(llvm::BitstreamCursor &Cursor)
: Cursor(Cursor), Offset(Cursor.GetCurrentBitNo()) {}

~SavedStreamPosition() {
Cursor.JumpToBit(Offset);
}

private:
llvm::BitstreamCursor &Cursor;
uint64_t Offset;
};

inline void PCHValidator::Error(const char *Msg) {
Reader.Error(Msg);
}

} 

#endif 
