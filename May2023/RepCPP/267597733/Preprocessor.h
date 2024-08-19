
#ifndef LLVM_CLANG_LEX_PREPROCESSOR_H
#define LLVM_CLANG_LEX_PREPROCESSOR_H

#include "clang/Basic/Builtins.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/ModuleLoader.h"
#include "clang/Lex/ModuleMap.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/PTHLexer.h"
#include "clang/Lex/Token.h"
#include "clang/Lex/TokenLexer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Registry.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace llvm {

template<unsigned InternalLen> class SmallString;

} 

namespace clang {

class CodeCompletionHandler;
class CommentHandler;
class DirectoryEntry;
class DirectoryLookup;
class ExternalPreprocessorSource;
class FileEntry;
class FileManager;
class HeaderSearch;
class MacroArgs;
class MemoryBufferCache;
class PragmaHandler;
class PragmaNamespace;
class PreprocessingRecord;
class PreprocessorLexer;
class PreprocessorOptions;
class PTHManager;
class ScratchBuffer;
class TargetInfo;

class TokenValue {
tok::TokenKind Kind;
IdentifierInfo *II;

public:
TokenValue(tok::TokenKind Kind) : Kind(Kind), II(nullptr) {
assert(Kind != tok::raw_identifier && "Raw identifiers are not supported.");
assert(Kind != tok::identifier &&
"Identifiers should be created by TokenValue(IdentifierInfo *)");
assert(!tok::isLiteral(Kind) && "Literals are not supported.");
assert(!tok::isAnnotation(Kind) && "Annotations are not supported.");
}

TokenValue(IdentifierInfo *II) : Kind(tok::identifier), II(II) {}

bool operator==(const Token &Tok) const {
return Tok.getKind() == Kind &&
(!II || II == Tok.getIdentifierInfo());
}
};

enum MacroUse {
MU_Other  = 0,

MU_Define = 1,

MU_Undef  = 2
};

class Preprocessor {
friend class VAOptDefinitionContext;
friend class VariadicMacroScopeGuard;

std::shared_ptr<PreprocessorOptions> PPOpts;
DiagnosticsEngine        *Diags;
LangOptions       &LangOpts;
const TargetInfo *Target = nullptr;
const TargetInfo *AuxTarget = nullptr;
FileManager       &FileMgr;
SourceManager     &SourceMgr;
MemoryBufferCache &PCMCache;
std::unique_ptr<ScratchBuffer> ScratchBuf;
HeaderSearch      &HeaderInfo;
ModuleLoader      &TheModuleLoader;

ExternalPreprocessorSource *ExternalSource;

std::unique_ptr<PTHManager> PTH;

llvm::BumpPtrAllocator BP;

IdentifierInfo *Ident__LINE__, *Ident__FILE__;   
IdentifierInfo *Ident__DATE__, *Ident__TIME__;   
IdentifierInfo *Ident__INCLUDE_LEVEL__;          
IdentifierInfo *Ident__BASE_FILE__;              
IdentifierInfo *Ident__TIMESTAMP__;              
IdentifierInfo *Ident__COUNTER__;                
IdentifierInfo *Ident_Pragma, *Ident__pragma;    
IdentifierInfo *Ident__identifier;               
IdentifierInfo *Ident__VA_ARGS__;                
IdentifierInfo *Ident__VA_OPT__;                 
IdentifierInfo *Ident__has_feature;              
IdentifierInfo *Ident__has_extension;            
IdentifierInfo *Ident__has_builtin;              
IdentifierInfo *Ident__has_attribute;            
IdentifierInfo *Ident__has_include;              
IdentifierInfo *Ident__has_include_next;         
IdentifierInfo *Ident__has_warning;              
IdentifierInfo *Ident__is_identifier;            
IdentifierInfo *Ident__building_module;          
IdentifierInfo *Ident__MODULE__;                 
IdentifierInfo *Ident__has_cpp_attribute;        
IdentifierInfo *Ident__has_c_attribute;          
IdentifierInfo *Ident__has_declspec;             
IdentifierInfo *Ident__is_target_arch;           
IdentifierInfo *Ident__is_target_vendor;         
IdentifierInfo *Ident__is_target_os;             
IdentifierInfo *Ident__is_target_environment;    

SourceLocation DATELoc, TIMELoc;

unsigned CounterValue = 0;

enum {
MaxAllowedIncludeStackDepth = 200
};

bool KeepComments : 1;
bool KeepMacroComments : 1;
bool SuppressIncludeNotFoundError : 1;

bool InMacroArgs : 1;            

bool OwnsHeaderSearch : 1;

bool DisableMacroExpansion : 1;

bool MacroExpansionInDirectivesOverride : 1;

class ResetMacroExpansionHelper;

mutable bool ReadMacrosFromExternalSource : 1;

bool PragmasEnabled : 1;

bool PreprocessedOutput : 1;

bool ParsingIfOrElifDirective;

bool InMacroArgPreExpansion;

mutable IdentifierTable Identifiers;

SelectorTable Selectors;

Builtin::Context BuiltinInfo;

std::unique_ptr<PragmaNamespace> PragmaHandlers;

std::unique_ptr<PragmaNamespace> PragmaHandlersBackup;

std::vector<CommentHandler *> CommentHandlers;

bool IncrementalProcessing = false;

TranslationUnitKind TUKind;

CodeCompletionHandler *CodeComplete = nullptr;

const FileEntry *CodeCompletionFile = nullptr;

unsigned CodeCompletionOffset = 0;

SourceLocation CodeCompletionLoc;

SourceLocation CodeCompletionFileLoc;

SourceLocation ModuleImportLoc;

SmallVector<std::pair<IdentifierInfo *, SourceLocation>, 2> ModuleImportPath;

bool LastTokenWasAt = false;

bool ModuleImportExpectsIdentifier = false;

SourceLocation PragmaARCCFCodeAuditedLoc;

SourceLocation PragmaAssumeNonNullLoc;

bool CodeCompletionReached = false;

IdentifierInfo *CodeCompletionII = nullptr;

const DirectoryEntry *MainFileDir = nullptr;

std::pair<int, bool> SkipMainFilePreamble;

public:
struct PreambleSkipInfo {
SourceLocation HashTokenLoc;
SourceLocation IfTokenLoc;
bool FoundNonSkipPortion;
bool FoundElse;
SourceLocation ElseLoc;

PreambleSkipInfo(SourceLocation HashTokenLoc, SourceLocation IfTokenLoc,
bool FoundNonSkipPortion, bool FoundElse,
SourceLocation ElseLoc)
: HashTokenLoc(HashTokenLoc), IfTokenLoc(IfTokenLoc),
FoundNonSkipPortion(FoundNonSkipPortion), FoundElse(FoundElse),
ElseLoc(ElseLoc) {}
};

private:
friend class ASTReader;
friend class MacroArgs;

class PreambleConditionalStackStore {
enum State {
Off = 0,
Recording = 1,
Replaying = 2,
};

public:
PreambleConditionalStackStore() = default;

void startRecording() { ConditionalStackState = Recording; }
void startReplaying() { ConditionalStackState = Replaying; }
bool isRecording() const { return ConditionalStackState == Recording; }
bool isReplaying() const { return ConditionalStackState == Replaying; }

ArrayRef<PPConditionalInfo> getStack() const {
return ConditionalStack;
}

void doneReplaying() {
ConditionalStack.clear();
ConditionalStackState = Off;
}

void setStack(ArrayRef<PPConditionalInfo> s) {
if (!isRecording() && !isReplaying())
return;
ConditionalStack.clear();
ConditionalStack.append(s.begin(), s.end());
}

bool hasRecordedPreamble() const { return !ConditionalStack.empty(); }

bool reachedEOFWhileSkipping() const { return SkipInfo.hasValue(); }

void clearSkipInfo() { SkipInfo.reset(); }

llvm::Optional<PreambleSkipInfo> SkipInfo;

private:
SmallVector<PPConditionalInfo, 4> ConditionalStack;
State ConditionalStackState = Off;
} PreambleConditionalStack;

std::unique_ptr<Lexer> CurLexer;

std::unique_ptr<PTHLexer> CurPTHLexer;

PreprocessorLexer *CurPPLexer = nullptr;

const DirectoryLookup *CurDirLookup = nullptr;

std::unique_ptr<TokenLexer> CurTokenLexer;

enum CurLexerKind {
CLK_Lexer,
CLK_PTHLexer,
CLK_TokenLexer,
CLK_CachingLexer,
CLK_LexAfterModuleImport
} CurLexerKind = CLK_Lexer;

Module *CurLexerSubmodule = nullptr;

struct IncludeStackInfo {
enum CurLexerKind           CurLexerKind;
Module                     *TheSubmodule;
std::unique_ptr<Lexer>      TheLexer;
std::unique_ptr<PTHLexer>   ThePTHLexer;
PreprocessorLexer          *ThePPLexer;
std::unique_ptr<TokenLexer> TheTokenLexer;
const DirectoryLookup      *TheDirLookup;

IncludeStackInfo(enum CurLexerKind CurLexerKind, Module *TheSubmodule,
std::unique_ptr<Lexer> &&TheLexer,
std::unique_ptr<PTHLexer> &&ThePTHLexer,
PreprocessorLexer *ThePPLexer,
std::unique_ptr<TokenLexer> &&TheTokenLexer,
const DirectoryLookup *TheDirLookup)
: CurLexerKind(std::move(CurLexerKind)),
TheSubmodule(std::move(TheSubmodule)), TheLexer(std::move(TheLexer)),
ThePTHLexer(std::move(ThePTHLexer)),
ThePPLexer(std::move(ThePPLexer)),
TheTokenLexer(std::move(TheTokenLexer)),
TheDirLookup(std::move(TheDirLookup)) {}
};
std::vector<IncludeStackInfo> IncludeMacroStack;

std::unique_ptr<PPCallbacks> Callbacks;

struct MacroExpandsInfo {
Token Tok;
MacroDefinition MD;
SourceRange Range;

MacroExpandsInfo(Token Tok, MacroDefinition MD, SourceRange Range)
: Tok(Tok), MD(MD), Range(Range) {}
};
SmallVector<MacroExpandsInfo, 2> DelayedMacroExpandsCallbacks;

struct ModuleMacroInfo {
MacroDirective *MD;

llvm::TinyPtrVector<ModuleMacro *> ActiveModuleMacros;

unsigned ActiveModuleMacrosGeneration = 0;

bool IsAmbiguous = false;

llvm::TinyPtrVector<ModuleMacro *> OverriddenMacros;

ModuleMacroInfo(MacroDirective *MD) : MD(MD) {}
};

class MacroState {
mutable llvm::PointerUnion<MacroDirective *, ModuleMacroInfo *> State;

ModuleMacroInfo *getModuleInfo(Preprocessor &PP,
const IdentifierInfo *II) const {
if (II->isOutOfDate())
PP.updateOutOfDateIdentifier(const_cast<IdentifierInfo&>(*II));
if (!II->hasMacroDefinition() ||
(!PP.getLangOpts().Modules &&
!PP.getLangOpts().ModulesLocalVisibility) ||
!PP.CurSubmoduleState->VisibleModules.getGeneration())
return nullptr;

auto *Info = State.dyn_cast<ModuleMacroInfo*>();
if (!Info) {
Info = new (PP.getPreprocessorAllocator())
ModuleMacroInfo(State.get<MacroDirective *>());
State = Info;
}

if (PP.CurSubmoduleState->VisibleModules.getGeneration() !=
Info->ActiveModuleMacrosGeneration)
PP.updateModuleMacroInfo(II, *Info);
return Info;
}

public:
MacroState() : MacroState(nullptr) {}
MacroState(MacroDirective *MD) : State(MD) {}

MacroState(MacroState &&O) noexcept : State(O.State) {
O.State = (MacroDirective *)nullptr;
}

MacroState &operator=(MacroState &&O) noexcept {
auto S = O.State;
O.State = (MacroDirective *)nullptr;
State = S;
return *this;
}

~MacroState() {
if (auto *Info = State.dyn_cast<ModuleMacroInfo*>())
Info->~ModuleMacroInfo();
}

MacroDirective *getLatest() const {
if (auto *Info = State.dyn_cast<ModuleMacroInfo*>())
return Info->MD;
return State.get<MacroDirective*>();
}

void setLatest(MacroDirective *MD) {
if (auto *Info = State.dyn_cast<ModuleMacroInfo*>())
Info->MD = MD;
else
State = MD;
}

bool isAmbiguous(Preprocessor &PP, const IdentifierInfo *II) const {
auto *Info = getModuleInfo(PP, II);
return Info ? Info->IsAmbiguous : false;
}

ArrayRef<ModuleMacro *>
getActiveModuleMacros(Preprocessor &PP, const IdentifierInfo *II) const {
if (auto *Info = getModuleInfo(PP, II))
return Info->ActiveModuleMacros;
return None;
}

MacroDirective::DefInfo findDirectiveAtLoc(SourceLocation Loc,
SourceManager &SourceMgr) const {
if (auto *Latest = getLatest())
return Latest->findDirectiveAtLoc(Loc, SourceMgr);
return {};
}

void overrideActiveModuleMacros(Preprocessor &PP, IdentifierInfo *II) {
if (auto *Info = getModuleInfo(PP, II)) {
Info->OverriddenMacros.insert(Info->OverriddenMacros.end(),
Info->ActiveModuleMacros.begin(),
Info->ActiveModuleMacros.end());
Info->ActiveModuleMacros.clear();
Info->IsAmbiguous = false;
}
}

ArrayRef<ModuleMacro*> getOverriddenMacros() const {
if (auto *Info = State.dyn_cast<ModuleMacroInfo*>())
return Info->OverriddenMacros;
return None;
}

void setOverriddenMacros(Preprocessor &PP,
ArrayRef<ModuleMacro *> Overrides) {
auto *Info = State.dyn_cast<ModuleMacroInfo*>();
if (!Info) {
if (Overrides.empty())
return;
Info = new (PP.getPreprocessorAllocator())
ModuleMacroInfo(State.get<MacroDirective *>());
State = Info;
}
Info->OverriddenMacros.clear();
Info->OverriddenMacros.insert(Info->OverriddenMacros.end(),
Overrides.begin(), Overrides.end());
Info->ActiveModuleMacrosGeneration = 0;
}
};

using MacroMap = llvm::DenseMap<const IdentifierInfo *, MacroState>;

struct SubmoduleState;

struct BuildingSubmoduleInfo {
Module *M;

SourceLocation ImportLoc;

bool IsPragma;

SubmoduleState *OuterSubmoduleState;

unsigned OuterPendingModuleMacroNames;

BuildingSubmoduleInfo(Module *M, SourceLocation ImportLoc, bool IsPragma,
SubmoduleState *OuterSubmoduleState,
unsigned OuterPendingModuleMacroNames)
: M(M), ImportLoc(ImportLoc), IsPragma(IsPragma),
OuterSubmoduleState(OuterSubmoduleState),
OuterPendingModuleMacroNames(OuterPendingModuleMacroNames) {}
};
SmallVector<BuildingSubmoduleInfo, 8> BuildingSubmoduleStack;

struct SubmoduleState {
MacroMap Macros;

VisibleModuleSet VisibleModules;

};
std::map<Module *, SubmoduleState> Submodules;

SubmoduleState NullSubmoduleState;

SubmoduleState *CurSubmoduleState;

llvm::FoldingSet<ModuleMacro> ModuleMacros;

llvm::SmallVector<const IdentifierInfo *, 32> PendingModuleMacroNames;

llvm::DenseMap<const IdentifierInfo *, llvm::TinyPtrVector<ModuleMacro *>>
LeafModuleMacros;

using WarnUnusedMacroLocsTy = llvm::SmallPtrSet<SourceLocation, 32>;
WarnUnusedMacroLocsTy WarnUnusedMacroLocs;

MacroArgs *MacroArgCache = nullptr;

llvm::DenseMap<IdentifierInfo *, std::vector<MacroInfo *>>
PragmaPushMacroInfo;

unsigned NumDirectives = 0;
unsigned NumDefined = 0;
unsigned NumUndefined = 0;
unsigned NumPragma = 0;
unsigned NumIf = 0;
unsigned NumElse = 0;
unsigned NumEndif = 0;
unsigned NumEnteredSourceFiles = 0;
unsigned MaxIncludeStackDepth = 0;
unsigned NumMacroExpanded = 0;
unsigned NumFnMacroExpanded = 0;
unsigned NumBuiltinMacroExpanded = 0;
unsigned NumFastMacroExpanded = 0;
unsigned NumTokenPaste = 0;
unsigned NumFastTokenPaste = 0;
unsigned NumSkipped = 0;

std::string Predefines;

FileID PredefinesFileID;

FileID PCHThroughHeaderFileID;

bool SkippingUntilPCHThroughHeader = false;

enum { TokenLexerCacheSize = 8 };
unsigned NumCachedTokenLexers;
std::unique_ptr<TokenLexer> TokenLexerCache[TokenLexerCacheSize];

SmallVector<Token, 16> MacroExpandedTokens;
std::vector<std::pair<TokenLexer *, size_t>> MacroExpandingLexersStack;

PreprocessingRecord *Record = nullptr;

using CachedTokensTy = SmallVector<Token, 1>;

CachedTokensTy CachedTokens;

CachedTokensTy::size_type CachedLexPos = 0;

std::vector<CachedTokensTy::size_type> BacktrackPositions;

struct MacroInfoChain {
MacroInfo MI;
MacroInfoChain *Next;
};

MacroInfoChain *MIChainHead = nullptr;

void updateOutOfDateIdentifier(IdentifierInfo &II) const;

public:
Preprocessor(std::shared_ptr<PreprocessorOptions> PPOpts,
DiagnosticsEngine &diags, LangOptions &opts, SourceManager &SM,
MemoryBufferCache &PCMCache,
HeaderSearch &Headers, ModuleLoader &TheModuleLoader,
IdentifierInfoLookup *IILookup = nullptr,
bool OwnsHeaderSearch = false,
TranslationUnitKind TUKind = TU_Complete);

~Preprocessor();

void Initialize(const TargetInfo &Target,
const TargetInfo *AuxTarget = nullptr);

void InitializeForModelFile();

void FinalizeForModelFile();

PreprocessorOptions &getPreprocessorOpts() const { return *PPOpts; }

DiagnosticsEngine &getDiagnostics() const { return *Diags; }
void setDiagnostics(DiagnosticsEngine &D) { Diags = &D; }

const LangOptions &getLangOpts() const { return LangOpts; }
const TargetInfo &getTargetInfo() const { return *Target; }
const TargetInfo *getAuxTargetInfo() const { return AuxTarget; }
FileManager &getFileManager() const { return FileMgr; }
SourceManager &getSourceManager() const { return SourceMgr; }
MemoryBufferCache &getPCMCache() const { return PCMCache; }
HeaderSearch &getHeaderSearchInfo() const { return HeaderInfo; }

IdentifierTable &getIdentifierTable() { return Identifiers; }
const IdentifierTable &getIdentifierTable() const { return Identifiers; }
SelectorTable &getSelectorTable() { return Selectors; }
Builtin::Context &getBuiltinInfo() { return BuiltinInfo; }
llvm::BumpPtrAllocator &getPreprocessorAllocator() { return BP; }

void setPTHManager(PTHManager* pm);

PTHManager *getPTHManager() { return PTH.get(); }

void setExternalSource(ExternalPreprocessorSource *Source) {
ExternalSource = Source;
}

ExternalPreprocessorSource *getExternalSource() const {
return ExternalSource;
}

ModuleLoader &getModuleLoader() const { return TheModuleLoader; }

bool hadModuleLoaderFatalFailure() const {
return TheModuleLoader.HadFatalFailure;
}

bool isParsingIfOrElifDirective() const {
return ParsingIfOrElifDirective;
}

void SetCommentRetentionState(bool KeepComments, bool KeepMacroComments) {
this->KeepComments = KeepComments | KeepMacroComments;
this->KeepMacroComments = KeepMacroComments;
}

bool getCommentRetentionState() const { return KeepComments; }

void setPragmasEnabled(bool Enabled) { PragmasEnabled = Enabled; }
bool getPragmasEnabled() const { return PragmasEnabled; }

void SetSuppressIncludeNotFoundError(bool Suppress) {
SuppressIncludeNotFoundError = Suppress;
}

bool GetSuppressIncludeNotFoundError() {
return SuppressIncludeNotFoundError;
}

void setPreprocessedOutput(bool IsPreprocessedOutput) {
PreprocessedOutput = IsPreprocessedOutput;
}

bool isPreprocessedOutput() const { return PreprocessedOutput; }

bool isCurrentLexer(const PreprocessorLexer *L) const {
return CurPPLexer == L;
}

PreprocessorLexer *getCurrentLexer() const { return CurPPLexer; }

PreprocessorLexer *getCurrentFileLexer() const;

Module *getCurrentLexerSubmodule() const { return CurLexerSubmodule; }

FileID getPredefinesFileID() const { return PredefinesFileID; }

PPCallbacks *getPPCallbacks() const { return Callbacks.get(); }
void addPPCallbacks(std::unique_ptr<PPCallbacks> C) {
if (Callbacks)
C = llvm::make_unique<PPChainedCallbacks>(std::move(C),
std::move(Callbacks));
Callbacks = std::move(C);
}

bool isMacroDefined(StringRef Id) {
return isMacroDefined(&Identifiers.get(Id));
}
bool isMacroDefined(const IdentifierInfo *II) {
return II->hasMacroDefinition() &&
(!getLangOpts().Modules || (bool)getMacroDefinition(II));
}

bool isMacroDefinedInLocalModule(const IdentifierInfo *II, Module *M) {
if (!II->hasMacroDefinition())
return false;
auto I = Submodules.find(M);
if (I == Submodules.end())
return false;
auto J = I->second.Macros.find(II);
if (J == I->second.Macros.end())
return false;
auto *MD = J->second.getLatest();
return MD && MD->isDefined();
}

MacroDefinition getMacroDefinition(const IdentifierInfo *II) {
if (!II->hasMacroDefinition())
return {};

MacroState &S = CurSubmoduleState->Macros[II];
auto *MD = S.getLatest();
while (MD && isa<VisibilityMacroDirective>(MD))
MD = MD->getPrevious();
return MacroDefinition(dyn_cast_or_null<DefMacroDirective>(MD),
S.getActiveModuleMacros(*this, II),
S.isAmbiguous(*this, II));
}

MacroDefinition getMacroDefinitionAtLoc(const IdentifierInfo *II,
SourceLocation Loc) {
if (!II->hadMacroDefinition())
return {};

MacroState &S = CurSubmoduleState->Macros[II];
MacroDirective::DefInfo DI;
if (auto *MD = S.getLatest())
DI = MD->findDirectiveAtLoc(Loc, getSourceManager());
return MacroDefinition(DI.getDirective(),
S.getActiveModuleMacros(*this, II),
S.isAmbiguous(*this, II));
}

MacroDirective *getLocalMacroDirective(const IdentifierInfo *II) const {
if (!II->hasMacroDefinition())
return nullptr;

auto *MD = getLocalMacroDirectiveHistory(II);
if (!MD || MD->getDefinition().isUndefined())
return nullptr;

return MD;
}

const MacroInfo *getMacroInfo(const IdentifierInfo *II) const {
return const_cast<Preprocessor*>(this)->getMacroInfo(II);
}

MacroInfo *getMacroInfo(const IdentifierInfo *II) {
if (!II->hasMacroDefinition())
return nullptr;
if (auto MD = getMacroDefinition(II))
return MD.getMacroInfo();
return nullptr;
}

MacroDirective *getLocalMacroDirectiveHistory(const IdentifierInfo *II) const;

void appendMacroDirective(IdentifierInfo *II, MacroDirective *MD);
DefMacroDirective *appendDefMacroDirective(IdentifierInfo *II, MacroInfo *MI,
SourceLocation Loc) {
DefMacroDirective *MD = AllocateDefMacroDirective(MI, Loc);
appendMacroDirective(II, MD);
return MD;
}
DefMacroDirective *appendDefMacroDirective(IdentifierInfo *II,
MacroInfo *MI) {
return appendDefMacroDirective(II, MI, MI->getDefinitionLoc());
}

void setLoadedMacroDirective(IdentifierInfo *II, MacroDirective *ED,
MacroDirective *MD);

ModuleMacro *addModuleMacro(Module *Mod, IdentifierInfo *II, MacroInfo *Macro,
ArrayRef<ModuleMacro *> Overrides, bool &IsNew);
ModuleMacro *getModuleMacro(Module *Mod, IdentifierInfo *II);

ArrayRef<ModuleMacro*> getLeafModuleMacros(const IdentifierInfo *II) const {
if (II->isOutOfDate())
updateOutOfDateIdentifier(const_cast<IdentifierInfo&>(*II));
auto I = LeafModuleMacros.find(II);
if (I != LeafModuleMacros.end())
return I->second;
return None;
}

using macro_iterator = MacroMap::const_iterator;

macro_iterator macro_begin(bool IncludeExternalMacros = true) const;
macro_iterator macro_end(bool IncludeExternalMacros = true) const;

llvm::iterator_range<macro_iterator>
macros(bool IncludeExternalMacros = true) const {
macro_iterator begin = macro_begin(IncludeExternalMacros);
macro_iterator end = macro_end(IncludeExternalMacros);
return llvm::make_range(begin, end);
}


StringRef getLastMacroWithSpelling(SourceLocation Loc,
ArrayRef<TokenValue> Tokens) const;

const std::string &getPredefines() const { return Predefines; }

void setPredefines(const char *P) { Predefines = P; }
void setPredefines(StringRef P) { Predefines = P; }

IdentifierInfo *getIdentifierInfo(StringRef Name) const {
return &Identifiers.get(Name);
}

void AddPragmaHandler(StringRef Namespace, PragmaHandler *Handler);
void AddPragmaHandler(PragmaHandler *Handler) {
AddPragmaHandler(StringRef(), Handler);
}

void RemovePragmaHandler(StringRef Namespace, PragmaHandler *Handler);
void RemovePragmaHandler(PragmaHandler *Handler) {
RemovePragmaHandler(StringRef(), Handler);
}

void IgnorePragmas();

void addCommentHandler(CommentHandler *Handler);

void removeCommentHandler(CommentHandler *Handler);

void setCodeCompletionHandler(CodeCompletionHandler &Handler) {
CodeComplete = &Handler;
}

CodeCompletionHandler *getCodeCompletionHandler() const {
return CodeComplete;
}

void clearCodeCompletionHandler() {
CodeComplete = nullptr;
}

void CodeCompleteNaturalLanguage();

void setCodeCompletionIdentifierInfo(IdentifierInfo *Filter) {
CodeCompletionII = Filter;
}

StringRef getCodeCompletionFilter() {
if (CodeCompletionII)
return CodeCompletionII->getName();
return {};
}

PreprocessingRecord *getPreprocessingRecord() const { return Record; }

void createPreprocessingRecord();

bool isPCHThroughHeader(const FileEntry *File);

bool creatingPCHWithThroughHeader();

bool usingPCHWithThroughHeader();

void SkipTokensUntilPCHThroughHeader();

void HandleSkippedThroughHeaderDirective(Token &Result,
SourceLocation HashLoc);

void EnterMainSourceFile();

void EndSourceFile();

bool EnterSourceFile(FileID CurFileID, const DirectoryLookup *Dir,
SourceLocation Loc);

void EnterMacro(Token &Identifier, SourceLocation ILEnd, MacroInfo *Macro,
MacroArgs *Args);

private:
void EnterTokenStream(const Token *Toks, unsigned NumToks,
bool DisableMacroExpansion, bool OwnsTokens);

public:
void EnterTokenStream(std::unique_ptr<Token[]> Toks, unsigned NumToks,
bool DisableMacroExpansion) {
EnterTokenStream(Toks.release(), NumToks, DisableMacroExpansion, true);
}

void EnterTokenStream(ArrayRef<Token> Toks, bool DisableMacroExpansion) {
EnterTokenStream(Toks.data(), Toks.size(), DisableMacroExpansion, false);
}

void RemoveTopOfLexerStack();

void EnableBacktrackAtThisPos();

void CommitBacktrackedTokens();

struct CachedTokensRange {
CachedTokensTy::size_type Begin, End;
};

private:
Optional<CachedTokensRange> CachedTokenRangeToErase;

public:
CachedTokensRange LastCachedTokenRange();

void EraseCachedTokens(CachedTokensRange TokenRange);

void Backtrack();

bool isBacktrackEnabled() const { return !BacktrackPositions.empty(); }

void Lex(Token &Result);

void LexAfterModuleImport(Token &Result);

void makeModuleVisible(Module *M, SourceLocation Loc);

SourceLocation getModuleImportLoc(Module *M) const {
return CurSubmoduleState->VisibleModules.getImportLoc(M);
}

bool LexStringLiteral(Token &Result, std::string &String,
const char *DiagnosticTag, bool AllowMacroExpansion) {
if (AllowMacroExpansion)
Lex(Result);
else
LexUnexpandedToken(Result);
return FinishLexStringLiteral(Result, String, DiagnosticTag,
AllowMacroExpansion);
}

bool FinishLexStringLiteral(Token &Result, std::string &String,
const char *DiagnosticTag,
bool AllowMacroExpansion);

void LexNonComment(Token &Result) {
do
Lex(Result);
while (Result.getKind() == tok::comment);
}

void LexUnexpandedToken(Token &Result) {
bool OldVal = DisableMacroExpansion;
DisableMacroExpansion = true;
Lex(Result);

DisableMacroExpansion = OldVal;
}

void LexUnexpandedNonComment(Token &Result) {
do
LexUnexpandedToken(Result);
while (Result.getKind() == tok::comment);
}

bool parseSimpleIntegerLiteral(Token &Tok, uint64_t &Value);

void SetMacroExpansionOnlyInDirectives() {
DisableMacroExpansion = true;
MacroExpansionInDirectivesOverride = true;
}

const Token &LookAhead(unsigned N) {
if (CachedLexPos + N < CachedTokens.size())
return CachedTokens[CachedLexPos+N];
else
return PeekAhead(N+1);
}

void RevertCachedTokens(unsigned N) {
assert(isBacktrackEnabled() &&
"Should only be called when tokens are cached for backtracking");
assert(signed(CachedLexPos) - signed(N) >= signed(BacktrackPositions.back())
&& "Should revert tokens up to the last backtrack position, not more");
assert(signed(CachedLexPos) - signed(N) >= 0 &&
"Corrupted backtrack positions ?");
CachedLexPos -= N;
}

void EnterToken(const Token &Tok) {
EnterCachingLexMode();
CachedTokens.insert(CachedTokens.begin()+CachedLexPos, Tok);
}

void AnnotateCachedTokens(const Token &Tok) {
assert(Tok.isAnnotation() && "Expected annotation token");
if (CachedLexPos != 0 && isBacktrackEnabled())
AnnotatePreviousCachedTokens(Tok);
}

SourceLocation getLastCachedTokenLocation() const {
assert(CachedLexPos != 0);
return CachedTokens[CachedLexPos-1].getLastLoc();
}

bool IsPreviousCachedToken(const Token &Tok) const;

void ReplacePreviousCachedToken(ArrayRef<Token> NewToks);

void ReplaceLastTokenWithAnnotation(const Token &Tok) {
assert(Tok.isAnnotation() && "Expected annotation token");
if (CachedLexPos != 0 && isBacktrackEnabled())
CachedTokens[CachedLexPos-1] = Tok;
}

void EnterAnnotationToken(SourceRange Range, tok::TokenKind Kind,
void *AnnotationVal);

void TypoCorrectToken(const Token &Tok) {
assert(Tok.getIdentifierInfo() && "Expected identifier token");
if (CachedLexPos != 0 && isBacktrackEnabled())
CachedTokens[CachedLexPos-1] = Tok;
}

void recomputeCurLexerKind();

bool isIncrementalProcessingEnabled() const { return IncrementalProcessing; }

void enableIncrementalProcessing(bool value = true) {
IncrementalProcessing = value;
}

bool SetCodeCompletionPoint(const FileEntry *File,
unsigned Line, unsigned Column);

bool isCodeCompletionEnabled() const { return CodeCompletionFile != nullptr; }

SourceLocation getCodeCompletionLoc() const { return CodeCompletionLoc; }

SourceLocation getCodeCompletionFileLoc() const {
return CodeCompletionFileLoc;
}

bool isCodeCompletionReached() const { return CodeCompletionReached; }

void setCodeCompletionReached() {
assert(isCodeCompletionEnabled() && "Code-completion not enabled!");
CodeCompletionReached = true;
getDiagnostics().setSuppressAllDiagnostics(true);
}

SourceLocation getPragmaARCCFCodeAuditedLoc() const {
return PragmaARCCFCodeAuditedLoc;
}

void setPragmaARCCFCodeAuditedLoc(SourceLocation Loc) {
PragmaARCCFCodeAuditedLoc = Loc;
}

SourceLocation getPragmaAssumeNonNullLoc() const {
return PragmaAssumeNonNullLoc;
}

void setPragmaAssumeNonNullLoc(SourceLocation Loc) {
PragmaAssumeNonNullLoc = Loc;
}

void setMainFileDir(const DirectoryEntry *Dir) {
MainFileDir = Dir;
}

void setSkipMainFilePreamble(unsigned Bytes, bool StartOfLine) {
SkipMainFilePreamble.first = Bytes;
SkipMainFilePreamble.second = StartOfLine;
}

DiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID) const {
return Diags->Report(Loc, DiagID);
}

DiagnosticBuilder Diag(const Token &Tok, unsigned DiagID) const {
return Diags->Report(Tok.getLocation(), DiagID);
}

StringRef getSpelling(SourceLocation loc,
SmallVectorImpl<char> &buffer,
bool *invalid = nullptr) const {
return Lexer::getSpelling(loc, buffer, SourceMgr, LangOpts, invalid);
}

std::string getSpelling(const Token &Tok, bool *Invalid = nullptr) const {
return Lexer::getSpelling(Tok, SourceMgr, LangOpts, Invalid);
}

unsigned getSpelling(const Token &Tok, const char *&Buffer,
bool *Invalid = nullptr) const {
return Lexer::getSpelling(Tok, Buffer, SourceMgr, LangOpts, Invalid);
}

StringRef getSpelling(const Token &Tok,
SmallVectorImpl<char> &Buffer,
bool *Invalid = nullptr) const;

bool getRawToken(SourceLocation Loc, Token &Result,
bool IgnoreWhiteSpace = false) {
return Lexer::getRawToken(Loc, Result, SourceMgr, LangOpts, IgnoreWhiteSpace);
}

char
getSpellingOfSingleCharacterNumericConstant(const Token &Tok,
bool *Invalid = nullptr) const {
assert(Tok.is(tok::numeric_constant) &&
Tok.getLength() == 1 && "Called on unsupported token");
assert(!Tok.needsCleaning() && "Token can't need cleaning with length 1");

if (const char *D = Tok.getLiteralData())
return *D;

return *SourceMgr.getCharacterData(Tok.getLocation(), Invalid);
}

StringRef getImmediateMacroName(SourceLocation Loc) {
return Lexer::getImmediateMacroName(Loc, SourceMgr, getLangOpts());
}

void CreateString(StringRef Str, Token &Tok,
SourceLocation ExpansionLocStart = SourceLocation(),
SourceLocation ExpansionLocEnd = SourceLocation());

SourceLocation SplitToken(SourceLocation TokLoc, unsigned Length);

SourceLocation getLocForEndOfToken(SourceLocation Loc, unsigned Offset = 0) {
return Lexer::getLocForEndOfToken(Loc, Offset, SourceMgr, LangOpts);
}

bool isAtStartOfMacroExpansion(SourceLocation loc,
SourceLocation *MacroBegin = nullptr) const {
return Lexer::isAtStartOfMacroExpansion(loc, SourceMgr, LangOpts,
MacroBegin);
}

bool isAtEndOfMacroExpansion(SourceLocation loc,
SourceLocation *MacroEnd = nullptr) const {
return Lexer::isAtEndOfMacroExpansion(loc, SourceMgr, LangOpts, MacroEnd);
}

void DumpToken(const Token &Tok, bool DumpFlags = false) const;
void DumpLocation(SourceLocation Loc) const;
void DumpMacro(const MacroInfo &MI) const;
void dumpMacroInfo(const IdentifierInfo *II);

SourceLocation AdvanceToTokenCharacter(SourceLocation TokStart,
unsigned Char) const {
return Lexer::AdvanceToTokenCharacter(TokStart, Char, SourceMgr, LangOpts);
}

void IncrementPasteCounter(bool isFast) {
if (isFast)
++NumFastTokenPaste;
else
++NumTokenPaste;
}

void PrintStats();

size_t getTotalMemory() const;

void HandleMicrosoftCommentPaste(Token &Tok);


IdentifierInfo *LookUpIdentifierInfo(Token &Identifier) const;

private:
llvm::DenseMap<IdentifierInfo*,unsigned> PoisonReasons;

public:
void SetPoisonReason(IdentifierInfo *II, unsigned DiagID);

void HandlePoisonedIdentifier(Token & Tok);

void MaybeHandlePoisonedIdentifier(Token & Identifier) {
if(IdentifierInfo * II = Identifier.getIdentifierInfo()) {
if(II->isPoisoned()) {
HandlePoisonedIdentifier(Identifier);
}
}
}

private:
IdentifierInfo *Ident__exception_code,
*Ident___exception_code,
*Ident_GetExceptionCode;
IdentifierInfo *Ident__exception_info,
*Ident___exception_info,
*Ident_GetExceptionInfo;
IdentifierInfo *Ident__abnormal_termination,
*Ident___abnormal_termination,
*Ident_AbnormalTermination;

const char *getCurLexerEndPos();
void diagnoseMissingHeaderInUmbrellaDir(const Module &Mod);

public:
void PoisonSEHIdentifiers(bool Poison = true); 

bool HandleIdentifier(Token &Identifier);

bool HandleEndOfFile(Token &Result, bool isEndOfMacro = false);

bool HandleEndOfTokenLexer(Token &Result);

void HandleDirective(Token &Result);

void CheckEndOfDirective(const char *Directive, bool EnableMacros = false);

void DiscardUntilEndOfDirective();

bool SawDateOrTime() const {
return DATELoc != SourceLocation() || TIMELoc != SourceLocation();
}
unsigned getCounterValue() const { return CounterValue; }
void setCounterValue(unsigned V) { CounterValue = V; }

Module *getCurrentModule();

MacroInfo *AllocateMacroInfo(SourceLocation L);

bool GetIncludeFilenameSpelling(SourceLocation Loc,StringRef &Filename);

const FileEntry *LookupFile(SourceLocation FilenameLoc, StringRef Filename,
bool isAngled, const DirectoryLookup *FromDir,
const FileEntry *FromFile,
const DirectoryLookup *&CurDir,
SmallVectorImpl<char> *SearchPath,
SmallVectorImpl<char> *RelativePath,
ModuleMap::KnownHeader *SuggestedModule,
bool *IsMapped, bool SkipCache = false);

const DirectoryLookup *GetCurDirLookup() { return CurDirLookup; }

bool isInPrimaryFile() const;

bool ConcatenateIncludeName(SmallString<128> &FilenameBuffer,
SourceLocation &End);

bool LexOnOffSwitch(tok::OnOffSwitch &OOS);

bool CheckMacroName(Token &MacroNameTok, MacroUse isDefineUndef,
bool *ShadowFlag = nullptr);

void EnterSubmodule(Module *M, SourceLocation ImportLoc, bool ForPragma);
Module *LeaveSubmodule(bool ForPragma);

private:
friend void TokenLexer::ExpandFunctionArguments();

void PushIncludeMacroStack() {
assert(CurLexerKind != CLK_CachingLexer && "cannot push a caching lexer");
IncludeMacroStack.emplace_back(CurLexerKind, CurLexerSubmodule,
std::move(CurLexer), std::move(CurPTHLexer),
CurPPLexer, std::move(CurTokenLexer),
CurDirLookup);
CurPPLexer = nullptr;
}

void PopIncludeMacroStack() {
CurLexer = std::move(IncludeMacroStack.back().TheLexer);
CurPTHLexer = std::move(IncludeMacroStack.back().ThePTHLexer);
CurPPLexer = IncludeMacroStack.back().ThePPLexer;
CurTokenLexer = std::move(IncludeMacroStack.back().TheTokenLexer);
CurDirLookup  = IncludeMacroStack.back().TheDirLookup;
CurLexerSubmodule = IncludeMacroStack.back().TheSubmodule;
CurLexerKind = IncludeMacroStack.back().CurLexerKind;
IncludeMacroStack.pop_back();
}

void PropagateLineStartLeadingSpaceInfo(Token &Result);

bool needModuleMacros() const;

void updateModuleMacroInfo(const IdentifierInfo *II, ModuleMacroInfo &Info);

DefMacroDirective *AllocateDefMacroDirective(MacroInfo *MI,
SourceLocation Loc);
UndefMacroDirective *AllocateUndefMacroDirective(SourceLocation UndefLoc);
VisibilityMacroDirective *AllocateVisibilityMacroDirective(SourceLocation Loc,
bool isPublic);

void ReadMacroName(Token &MacroNameTok, MacroUse IsDefineUndef = MU_Other,
bool *ShadowFlag = nullptr);

MacroInfo *ReadOptionalMacroParameterListAndBody(
const Token &MacroNameTok, bool ImmediatelyAfterHeaderGuard);

bool ReadMacroParameterList(MacroInfo *MI, Token& LastTok);

void SkipExcludedConditionalBlock(SourceLocation HashTokenLoc,
SourceLocation IfTokenLoc,
bool FoundNonSkipPortion, bool FoundElse,
SourceLocation ElseLoc = SourceLocation());

void PTHSkipExcludedConditionalBlock();

struct DirectiveEvalResult {
bool Conditional;

bool IncludedUndefinedIds;
};

DirectiveEvalResult EvaluateDirectiveExpression(IdentifierInfo *&IfNDefMacro);

void RegisterBuiltinPragmas();

void RegisterBuiltinMacros();

bool HandleMacroExpandedIdentifier(Token &Tok, const MacroDefinition &MD);

Token *cacheMacroExpandedTokens(TokenLexer *tokLexer,
ArrayRef<Token> tokens);

void removeCachedMacroExpandedTokensOfLastLexer();

bool isNextPPTokenLParen();

MacroArgs *ReadMacroCallArgumentList(Token &MacroName, MacroInfo *MI,
SourceLocation &ExpansionEnd);

void ExpandBuiltinMacro(Token &Tok);

void Handle_Pragma(Token &Tok);

void HandleMicrosoft__pragma(Token &Tok);

void EnterSourceFileWithLexer(Lexer *TheLexer, const DirectoryLookup *Dir);

void EnterSourceFileWithPTH(PTHLexer *PL, const DirectoryLookup *Dir);

void setPredefinesFileID(FileID FID) {
assert(PredefinesFileID.isInvalid() && "PredefinesFileID already set!");
PredefinesFileID = FID;
}

void setPCHThroughHeaderFileID(FileID FID);

static bool IsFileLexer(const Lexer* L, const PreprocessorLexer* P) {
return L ? !L->isPragmaLexer() : P != nullptr;
}

static bool IsFileLexer(const IncludeStackInfo& I) {
return IsFileLexer(I.TheLexer.get(), I.ThePPLexer);
}

bool IsFileLexer() const {
return IsFileLexer(CurLexer.get(), CurPPLexer);
}

void CachingLex(Token &Result);

bool InCachingLexMode() const {
return !CurPPLexer && !CurTokenLexer && !CurPTHLexer &&
!IncludeMacroStack.empty();
}

void EnterCachingLexMode();

void ExitCachingLexMode() {
if (InCachingLexMode())
RemoveTopOfLexerStack();
}

const Token &PeekAhead(unsigned N);
void AnnotatePreviousCachedTokens(const Token &Tok);

void HandleLineDirective();
void HandleDigitDirective(Token &Tok);
void HandleUserDiagnosticDirective(Token &Tok, bool isWarning);
void HandleIdentSCCSDirective(Token &Tok);
void HandleMacroPublicDirective(Token &Tok);
void HandleMacroPrivateDirective();

void HandleIncludeDirective(SourceLocation HashLoc,
Token &Tok,
const DirectoryLookup *LookupFrom = nullptr,
const FileEntry *LookupFromFile = nullptr,
bool isImport = false);
void HandleIncludeNextDirective(SourceLocation HashLoc, Token &Tok);
void HandleIncludeMacrosDirective(SourceLocation HashLoc, Token &Tok);
void HandleImportDirective(SourceLocation HashLoc, Token &Tok);
void HandleMicrosoftImportDirective(Token &Tok);

public:
static bool checkModuleIsAvailable(const LangOptions &LangOpts,
const TargetInfo &TargetInfo,
DiagnosticsEngine &Diags, Module *M);

Module *getModuleForLocation(SourceLocation Loc);

const FileEntry *getModuleHeaderToIncludeForDiagnostics(SourceLocation IncLoc,
Module *M,
SourceLocation MLoc);

bool isRecordingPreamble() const {
return PreambleConditionalStack.isRecording();
}

bool hasRecordedPreamble() const {
return PreambleConditionalStack.hasRecordedPreamble();
}

ArrayRef<PPConditionalInfo> getPreambleConditionalStack() const {
return PreambleConditionalStack.getStack();
}

void setRecordedPreambleConditionalStack(ArrayRef<PPConditionalInfo> s) {
PreambleConditionalStack.setStack(s);
}

void setReplayablePreambleConditionalStack(ArrayRef<PPConditionalInfo> s,
llvm::Optional<PreambleSkipInfo> SkipInfo) {
PreambleConditionalStack.startReplaying();
PreambleConditionalStack.setStack(s);
PreambleConditionalStack.SkipInfo = SkipInfo;
}

llvm::Optional<PreambleSkipInfo> getPreambleSkipInfo() const {
return PreambleConditionalStack.SkipInfo;
}

private:
void replayPreambleConditionalStack();

void HandleDefineDirective(Token &Tok, bool ImmediatelyAfterTopLevelIfndef);
void HandleUndefDirective();

void HandleIfdefDirective(Token &Tok, const Token &HashToken,
bool isIfndef, bool ReadAnyTokensBeforeDirective);
void HandleIfDirective(Token &Tok, const Token &HashToken,
bool ReadAnyTokensBeforeDirective);
void HandleEndifDirective(Token &Tok);
void HandleElseDirective(Token &Tok, const Token &HashToken);
void HandleElifDirective(Token &Tok, const Token &HashToken);

void HandlePragmaDirective(SourceLocation IntroducerLoc,
PragmaIntroducerKind Introducer);

public:
void HandlePragmaOnce(Token &OnceTok);
void HandlePragmaMark();
void HandlePragmaPoison();
void HandlePragmaSystemHeader(Token &SysHeaderTok);
void HandlePragmaDependency(Token &DependencyTok);
void HandlePragmaPushMacro(Token &Tok);
void HandlePragmaPopMacro(Token &Tok);
void HandlePragmaIncludeAlias(Token &Tok);
void HandlePragmaModuleBuild(Token &Tok);
IdentifierInfo *ParsePragmaPushOrPopMacro(Token &Tok);

bool HandleComment(Token &Token, SourceRange Comment);

void markMacroAsUsed(MacroInfo *MI);
};

class CommentHandler {
public:
virtual ~CommentHandler();

virtual bool HandleComment(Preprocessor &PP, SourceRange Comment) = 0;
};

using PragmaHandlerRegistry = llvm::Registry<PragmaHandler>;

} 

#endif 
