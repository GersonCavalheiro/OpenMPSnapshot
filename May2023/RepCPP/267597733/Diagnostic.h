
#ifndef LLVM_CLANG_BASIC_DIAGNOSTIC_H
#define LLVM_CLANG_BASIC_DIAGNOSTIC_H

#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <cstdint>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace clang {

class DeclContext;
class DiagnosticBuilder;
class DiagnosticConsumer;
class IdentifierInfo;
class LangOptions;
class Preprocessor;
class SourceManager;
class StoredDiagnostic;

namespace tok {

enum TokenKind : unsigned short;

} 

class FixItHint {
public:
CharSourceRange RemoveRange;

CharSourceRange InsertFromRange;

std::string CodeToInsert;

bool BeforePreviousInsertions = false;

FixItHint() = default;

bool isNull() const {
return !RemoveRange.isValid();
}

static FixItHint CreateInsertion(SourceLocation InsertionLoc,
StringRef Code,
bool BeforePreviousInsertions = false) {
FixItHint Hint;
Hint.RemoveRange =
CharSourceRange::getCharRange(InsertionLoc, InsertionLoc);
Hint.CodeToInsert = Code;
Hint.BeforePreviousInsertions = BeforePreviousInsertions;
return Hint;
}

static FixItHint CreateInsertionFromRange(SourceLocation InsertionLoc,
CharSourceRange FromRange,
bool BeforePreviousInsertions = false) {
FixItHint Hint;
Hint.RemoveRange =
CharSourceRange::getCharRange(InsertionLoc, InsertionLoc);
Hint.InsertFromRange = FromRange;
Hint.BeforePreviousInsertions = BeforePreviousInsertions;
return Hint;
}

static FixItHint CreateRemoval(CharSourceRange RemoveRange) {
FixItHint Hint;
Hint.RemoveRange = RemoveRange;
return Hint;
}
static FixItHint CreateRemoval(SourceRange RemoveRange) {
return CreateRemoval(CharSourceRange::getTokenRange(RemoveRange));
}

static FixItHint CreateReplacement(CharSourceRange RemoveRange,
StringRef Code) {
FixItHint Hint;
Hint.RemoveRange = RemoveRange;
Hint.CodeToInsert = Code;
return Hint;
}

static FixItHint CreateReplacement(SourceRange RemoveRange,
StringRef Code) {
return CreateReplacement(CharSourceRange::getTokenRange(RemoveRange), Code);
}
};

class DiagnosticsEngine : public RefCountedBase<DiagnosticsEngine> {
public:
enum Level {
Ignored = DiagnosticIDs::Ignored,
Note = DiagnosticIDs::Note,
Remark = DiagnosticIDs::Remark,
Warning = DiagnosticIDs::Warning,
Error = DiagnosticIDs::Error,
Fatal = DiagnosticIDs::Fatal
};

enum ArgumentKind {
ak_std_string,

ak_c_string,

ak_sint,

ak_uint,

ak_tokenkind,

ak_identifierinfo,

ak_qualtype,

ak_declarationname,

ak_nameddecl,

ak_nestednamespec,

ak_declcontext,

ak_qualtype_pair,

ak_attr
};

using ArgumentValue = std::pair<ArgumentKind, intptr_t>;

private:
unsigned char AllExtensionsSilenced = 0;

bool SuppressAfterFatalError = true;

bool SuppressAllDiagnostics = false;

bool ElideType = true;

bool PrintTemplateTree = false;

bool ShowColors = false;

OverloadsShown ShowOverloads = Ovl_All;

unsigned ErrorLimit = 0;

unsigned TemplateBacktraceLimit = 0;

unsigned ConstexprBacktraceLimit = 0;

IntrusiveRefCntPtr<DiagnosticIDs> Diags;
IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts;
DiagnosticConsumer *Client = nullptr;
std::unique_ptr<DiagnosticConsumer> Owner;
SourceManager *SourceMgr = nullptr;

class DiagState {
llvm::DenseMap<unsigned, DiagnosticMapping> DiagMap;

public:

unsigned IgnoreAllWarnings : 1;

unsigned EnableAllWarnings : 1;

unsigned WarningsAsErrors : 1;

unsigned ErrorsAsFatal : 1;

unsigned SuppressSystemWarnings : 1;

diag::Severity ExtBehavior = diag::Severity::Ignored;

DiagState()
: IgnoreAllWarnings(false), EnableAllWarnings(false),
WarningsAsErrors(false), ErrorsAsFatal(false),
SuppressSystemWarnings(false) {}

using iterator = llvm::DenseMap<unsigned, DiagnosticMapping>::iterator;
using const_iterator =
llvm::DenseMap<unsigned, DiagnosticMapping>::const_iterator;

void setMapping(diag::kind Diag, DiagnosticMapping Info) {
DiagMap[Diag] = Info;
}

DiagnosticMapping lookupMapping(diag::kind Diag) const {
return DiagMap.lookup(Diag);
}

DiagnosticMapping &getOrAddMapping(diag::kind Diag);

const_iterator begin() const { return DiagMap.begin(); }
const_iterator end() const { return DiagMap.end(); }
};

std::list<DiagState> DiagStates;

class DiagStateMap {
public:
void appendFirst(DiagState *State);

void append(SourceManager &SrcMgr, SourceLocation Loc, DiagState *State);

DiagState *lookup(SourceManager &SrcMgr, SourceLocation Loc) const;

bool empty() const { return Files.empty(); }

void clear() {
Files.clear();
FirstDiagState = CurDiagState = nullptr;
CurDiagStateLoc = SourceLocation();
}

LLVM_DUMP_METHOD void dump(SourceManager &SrcMgr,
StringRef DiagName = StringRef()) const;

DiagState *getCurDiagState() const { return CurDiagState; }

SourceLocation getCurDiagStateLoc() const { return CurDiagStateLoc; }

private:
friend class ASTReader;
friend class ASTWriter;

struct DiagStatePoint {
DiagState *State;
unsigned Offset;

DiagStatePoint(DiagState *State, unsigned Offset)
: State(State), Offset(Offset) {}
};

struct File {
File *Parent = nullptr;

unsigned ParentOffset = 0;

bool HasLocalTransitions = false;

llvm::SmallVector<DiagStatePoint, 4> StateTransitions;

DiagState *lookup(unsigned Offset) const;
};

mutable std::map<FileID, File> Files;

DiagState *FirstDiagState;

DiagState *CurDiagState;

SourceLocation CurDiagStateLoc;

File *getFile(SourceManager &SrcMgr, FileID ID) const;
};

DiagStateMap DiagStatesByLoc;

std::vector<DiagState *> DiagStateOnPushStack;

DiagState *GetCurDiagState() const {
return DiagStatesByLoc.getCurDiagState();
}

void PushDiagStatePoint(DiagState *State, SourceLocation L);

DiagState *GetDiagStateForLoc(SourceLocation Loc) const {
return SourceMgr ? DiagStatesByLoc.lookup(*SourceMgr, Loc)
: DiagStatesByLoc.getCurDiagState();
}

bool ErrorOccurred;

bool UncompilableErrorOccurred;

bool FatalErrorOccurred;

bool UnrecoverableErrorOccurred;

unsigned TrapNumErrorsOccurred;
unsigned TrapNumUnrecoverableErrorsOccurred;

DiagnosticIDs::Level LastDiagLevel;

unsigned NumWarnings;

unsigned NumErrors;

using ArgToStringFnTy = void (*)(
ArgumentKind Kind, intptr_t Val,
StringRef Modifier, StringRef Argument,
ArrayRef<ArgumentValue> PrevArgs,
SmallVectorImpl<char> &Output,
void *Cookie,
ArrayRef<intptr_t> QualTypeVals);

void *ArgToStringCookie = nullptr;
ArgToStringFnTy ArgToStringFn;

unsigned DelayedDiagID;

std::string DelayedDiagArg1;

std::string DelayedDiagArg2;

std::string FlagValue;

public:
explicit DiagnosticsEngine(IntrusiveRefCntPtr<DiagnosticIDs> Diags,
IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts,
DiagnosticConsumer *client = nullptr,
bool ShouldOwnClient = true);
DiagnosticsEngine(const DiagnosticsEngine &) = delete;
DiagnosticsEngine &operator=(const DiagnosticsEngine &) = delete;
~DiagnosticsEngine();

LLVM_DUMP_METHOD void dump() const { DiagStatesByLoc.dump(*SourceMgr); }
LLVM_DUMP_METHOD void dump(StringRef DiagName) const {
DiagStatesByLoc.dump(*SourceMgr, DiagName);
}

const IntrusiveRefCntPtr<DiagnosticIDs> &getDiagnosticIDs() const {
return Diags;
}

DiagnosticOptions &getDiagnosticOptions() const { return *DiagOpts; }

using diag_mapping_range = llvm::iterator_range<DiagState::const_iterator>;

diag_mapping_range getDiagnosticMappings() const {
const DiagState &DS = *GetCurDiagState();
return diag_mapping_range(DS.begin(), DS.end());
}

DiagnosticConsumer *getClient() { return Client; }
const DiagnosticConsumer *getClient() const { return Client; }

bool ownsClient() const { return Owner != nullptr; }

std::unique_ptr<DiagnosticConsumer> takeClient() { return std::move(Owner); }

bool hasSourceManager() const { return SourceMgr != nullptr; }

SourceManager &getSourceManager() const {
assert(SourceMgr && "SourceManager not set!");
return *SourceMgr;
}

void setSourceManager(SourceManager *SrcMgr) {
assert(DiagStatesByLoc.empty() &&
"Leftover diag state from a different SourceManager.");
SourceMgr = SrcMgr;
}


void pushMappings(SourceLocation Loc);

bool popMappings(SourceLocation Loc);

void setClient(DiagnosticConsumer *client, bool ShouldOwnClient = true);

void setErrorLimit(unsigned Limit) { ErrorLimit = Limit; }

void setTemplateBacktraceLimit(unsigned Limit) {
TemplateBacktraceLimit = Limit;
}

unsigned getTemplateBacktraceLimit() const {
return TemplateBacktraceLimit;
}

void setConstexprBacktraceLimit(unsigned Limit) {
ConstexprBacktraceLimit = Limit;
}

unsigned getConstexprBacktraceLimit() const {
return ConstexprBacktraceLimit;
}

void setIgnoreAllWarnings(bool Val) {
GetCurDiagState()->IgnoreAllWarnings = Val;
}
bool getIgnoreAllWarnings() const {
return GetCurDiagState()->IgnoreAllWarnings;
}

void setEnableAllWarnings(bool Val) {
GetCurDiagState()->EnableAllWarnings = Val;
}
bool getEnableAllWarnings() const {
return GetCurDiagState()->EnableAllWarnings;
}

void setWarningsAsErrors(bool Val) {
GetCurDiagState()->WarningsAsErrors = Val;
}
bool getWarningsAsErrors() const {
return GetCurDiagState()->WarningsAsErrors;
}

void setErrorsAsFatal(bool Val) { GetCurDiagState()->ErrorsAsFatal = Val; }
bool getErrorsAsFatal() const { return GetCurDiagState()->ErrorsAsFatal; }

void setSuppressAfterFatalError(bool Val) { SuppressAfterFatalError = Val; }

void setSuppressSystemWarnings(bool Val) {
GetCurDiagState()->SuppressSystemWarnings = Val;
}
bool getSuppressSystemWarnings() const {
return GetCurDiagState()->SuppressSystemWarnings;
}

void setSuppressAllDiagnostics(bool Val = true) {
SuppressAllDiagnostics = Val;
}
bool getSuppressAllDiagnostics() const { return SuppressAllDiagnostics; }

void setElideType(bool Val = true) { ElideType = Val; }
bool getElideType() { return ElideType; }

void setPrintTemplateTree(bool Val = false) { PrintTemplateTree = Val; }
bool getPrintTemplateTree() { return PrintTemplateTree; }

void setShowColors(bool Val = false) { ShowColors = Val; }
bool getShowColors() { return ShowColors; }

void setShowOverloads(OverloadsShown Val) {
ShowOverloads = Val;
}
OverloadsShown getShowOverloads() const { return ShowOverloads; }

void setLastDiagnosticIgnored(bool Ignored = true) {
if (LastDiagLevel == DiagnosticIDs::Fatal)
FatalErrorOccurred = true;
LastDiagLevel = Ignored ? DiagnosticIDs::Ignored : DiagnosticIDs::Warning;
}

bool isLastDiagnosticIgnored() const {
return LastDiagLevel == DiagnosticIDs::Ignored;
}

void setExtensionHandlingBehavior(diag::Severity H) {
GetCurDiagState()->ExtBehavior = H;
}
diag::Severity getExtensionHandlingBehavior() const {
return GetCurDiagState()->ExtBehavior;
}

void IncrementAllExtensionsSilenced() { ++AllExtensionsSilenced; }
void DecrementAllExtensionsSilenced() { --AllExtensionsSilenced; }
bool hasAllExtensionsSilenced() { return AllExtensionsSilenced != 0; }

void setSeverity(diag::kind Diag, diag::Severity Map, SourceLocation Loc);

bool setSeverityForGroup(diag::Flavor Flavor, StringRef Group,
diag::Severity Map,
SourceLocation Loc = SourceLocation());

bool setDiagnosticGroupWarningAsError(StringRef Group, bool Enabled);

bool setDiagnosticGroupErrorAsFatal(StringRef Group, bool Enabled);

void setSeverityForAll(diag::Flavor Flavor, diag::Severity Map,
SourceLocation Loc = SourceLocation());

bool hasErrorOccurred() const { return ErrorOccurred; }

bool hasUncompilableErrorOccurred() const {
return UncompilableErrorOccurred;
}
bool hasFatalErrorOccurred() const { return FatalErrorOccurred; }

bool hasUnrecoverableErrorOccurred() const {
return FatalErrorOccurred || UnrecoverableErrorOccurred;
}

unsigned getNumWarnings() const { return NumWarnings; }

void setNumWarnings(unsigned NumWarnings) {
this->NumWarnings = NumWarnings;
}

template <unsigned N>
unsigned getCustomDiagID(Level L, const char (&FormatString)[N]) {
return Diags->getCustomDiagID((DiagnosticIDs::Level)L,
StringRef(FormatString, N - 1));
}

void ConvertArgToString(ArgumentKind Kind, intptr_t Val,
StringRef Modifier, StringRef Argument,
ArrayRef<ArgumentValue> PrevArgs,
SmallVectorImpl<char> &Output,
ArrayRef<intptr_t> QualTypeVals) const {
ArgToStringFn(Kind, Val, Modifier, Argument, PrevArgs, Output,
ArgToStringCookie, QualTypeVals);
}

void SetArgToStringFn(ArgToStringFnTy Fn, void *Cookie) {
ArgToStringFn = Fn;
ArgToStringCookie = Cookie;
}

void notePriorDiagnosticFrom(const DiagnosticsEngine &Other) {
LastDiagLevel = Other.LastDiagLevel;
}

void Reset();


bool isIgnored(unsigned DiagID, SourceLocation Loc) const {
return Diags->getDiagnosticSeverity(DiagID, Loc, *this) ==
diag::Severity::Ignored;
}

Level getDiagnosticLevel(unsigned DiagID, SourceLocation Loc) const {
return (Level)Diags->getDiagnosticLevel(DiagID, Loc, *this);
}

inline DiagnosticBuilder Report(SourceLocation Loc, unsigned DiagID);
inline DiagnosticBuilder Report(unsigned DiagID);

void Report(const StoredDiagnostic &storedDiag);

bool isDiagnosticInFlight() const {
return CurDiagID != std::numeric_limits<unsigned>::max();
}

void SetDelayedDiagnostic(unsigned DiagID, StringRef Arg1 = "",
StringRef Arg2 = "");

void Clear() { CurDiagID = std::numeric_limits<unsigned>::max(); }

StringRef getFlagValue() const { return FlagValue; }

private:
friend class Diagnostic;
friend class DiagnosticBuilder;
friend class DiagnosticErrorTrap;
friend class DiagnosticIDs;
friend class PartialDiagnostic;

void ReportDelayed();

SourceLocation CurDiagLoc;

unsigned CurDiagID;

enum {
MaxArguments = 10,
};

signed char NumDiagArgs;

unsigned char DiagArgumentsKind[MaxArguments];

std::string DiagArgumentsStr[MaxArguments];

intptr_t DiagArgumentsVal[MaxArguments];

SmallVector<CharSourceRange, 8> DiagRanges;

SmallVector<FixItHint, 8> DiagFixItHints;

DiagnosticMapping makeUserMapping(diag::Severity Map, SourceLocation L) {
bool isPragma = L.isValid();
DiagnosticMapping Mapping =
DiagnosticMapping::Make(Map, true, isPragma);

if (isPragma) {
Mapping.setNoWarningAsError(true);
Mapping.setNoErrorAsFatal(true);
}

return Mapping;
}

bool ProcessDiag() {
return Diags->ProcessDiag(*this);
}

protected:
friend class ASTReader;
friend class ASTWriter;

friend class Sema;

bool EmitCurrentDiagnostic(bool Force = false);

unsigned getCurrentDiagID() const { return CurDiagID; }

SourceLocation getCurrentDiagLoc() const { return CurDiagLoc; }

};

class DiagnosticErrorTrap {
DiagnosticsEngine &Diag;
unsigned NumErrors;
unsigned NumUnrecoverableErrors;

public:
explicit DiagnosticErrorTrap(DiagnosticsEngine &Diag)
: Diag(Diag) { reset(); }

bool hasErrorOccurred() const {
return Diag.TrapNumErrorsOccurred > NumErrors;
}

bool hasUnrecoverableErrorOccurred() const {
return Diag.TrapNumUnrecoverableErrorsOccurred > NumUnrecoverableErrors;
}

void reset() {
NumErrors = Diag.TrapNumErrorsOccurred;
NumUnrecoverableErrors = Diag.TrapNumUnrecoverableErrorsOccurred;
}
};


class DiagnosticBuilder {
friend class DiagnosticsEngine;
friend class PartialDiagnostic;

mutable DiagnosticsEngine *DiagObj = nullptr;
mutable unsigned NumArgs = 0;

mutable bool IsActive = false;

mutable bool IsForceEmit = false;

DiagnosticBuilder() = default;

explicit DiagnosticBuilder(DiagnosticsEngine *diagObj)
: DiagObj(diagObj), IsActive(true) {
assert(diagObj && "DiagnosticBuilder requires a valid DiagnosticsEngine!");
diagObj->DiagRanges.clear();
diagObj->DiagFixItHints.clear();
}

protected:
void FlushCounts() {
DiagObj->NumDiagArgs = NumArgs;
}

void Clear() const {
DiagObj = nullptr;
IsActive = false;
IsForceEmit = false;
}

bool isActive() const { return IsActive; }

bool Emit() {
if (!isActive()) return false;

FlushCounts();

bool Result = DiagObj->EmitCurrentDiagnostic(IsForceEmit);

Clear();

return Result;
}

public:
DiagnosticBuilder(const DiagnosticBuilder &D) {
DiagObj = D.DiagObj;
IsActive = D.IsActive;
IsForceEmit = D.IsForceEmit;
D.Clear();
NumArgs = D.NumArgs;
}

DiagnosticBuilder &operator=(const DiagnosticBuilder &) = delete;

~DiagnosticBuilder() {
Emit();
}

static DiagnosticBuilder getEmpty() {
return {};
}

const DiagnosticBuilder &setForceEmit() const {
IsForceEmit = true;
return *this;
}

operator bool() const { return true; }

void AddString(StringRef S) const {
assert(isActive() && "Clients must not add to cleared diagnostic!");
assert(NumArgs < DiagnosticsEngine::MaxArguments &&
"Too many arguments to diagnostic!");
DiagObj->DiagArgumentsKind[NumArgs] = DiagnosticsEngine::ak_std_string;
DiagObj->DiagArgumentsStr[NumArgs++] = S;
}

void AddTaggedVal(intptr_t V, DiagnosticsEngine::ArgumentKind Kind) const {
assert(isActive() && "Clients must not add to cleared diagnostic!");
assert(NumArgs < DiagnosticsEngine::MaxArguments &&
"Too many arguments to diagnostic!");
DiagObj->DiagArgumentsKind[NumArgs] = Kind;
DiagObj->DiagArgumentsVal[NumArgs++] = V;
}

void AddSourceRange(const CharSourceRange &R) const {
assert(isActive() && "Clients must not add to cleared diagnostic!");
DiagObj->DiagRanges.push_back(R);
}

void AddFixItHint(const FixItHint &Hint) const {
assert(isActive() && "Clients must not add to cleared diagnostic!");
if (!Hint.isNull())
DiagObj->DiagFixItHints.push_back(Hint);
}

void addFlagValue(StringRef V) const { DiagObj->FlagValue = V; }
};

struct AddFlagValue {
StringRef Val;

explicit AddFlagValue(StringRef V) : Val(V) {}
};

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
const AddFlagValue V) {
DB.addFlagValue(V.Val);
return DB;
}

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
StringRef S) {
DB.AddString(S);
return DB;
}

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
const char *Str) {
DB.AddTaggedVal(reinterpret_cast<intptr_t>(Str),
DiagnosticsEngine::ak_c_string);
return DB;
}

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB, int I) {
DB.AddTaggedVal(I, DiagnosticsEngine::ak_sint);
return DB;
}

template <typename T>
inline
typename std::enable_if<std::is_same<T, bool>::value,
const DiagnosticBuilder &>::type
operator<<(const DiagnosticBuilder &DB, T I) {
DB.AddTaggedVal(I, DiagnosticsEngine::ak_sint);
return DB;
}

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
unsigned I) {
DB.AddTaggedVal(I, DiagnosticsEngine::ak_uint);
return DB;
}

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
tok::TokenKind I) {
DB.AddTaggedVal(static_cast<unsigned>(I), DiagnosticsEngine::ak_tokenkind);
return DB;
}

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
const IdentifierInfo *II) {
DB.AddTaggedVal(reinterpret_cast<intptr_t>(II),
DiagnosticsEngine::ak_identifierinfo);
return DB;
}

template <typename T>
inline typename std::enable_if<
std::is_same<typename std::remove_const<T>::type, DeclContext>::value,
const DiagnosticBuilder &>::type
operator<<(const DiagnosticBuilder &DB, T *DC) {
DB.AddTaggedVal(reinterpret_cast<intptr_t>(DC),
DiagnosticsEngine::ak_declcontext);
return DB;
}

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
SourceRange R) {
DB.AddSourceRange(CharSourceRange::getTokenRange(R));
return DB;
}

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
ArrayRef<SourceRange> Ranges) {
for (SourceRange R : Ranges)
DB.AddSourceRange(CharSourceRange::getTokenRange(R));
return DB;
}

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
const CharSourceRange &R) {
DB.AddSourceRange(R);
return DB;
}

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
const FixItHint &Hint) {
DB.AddFixItHint(Hint);
return DB;
}

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
ArrayRef<FixItHint> Hints) {
for (const FixItHint &Hint : Hints)
DB.AddFixItHint(Hint);
return DB;
}

using DiagNullabilityKind = std::pair<NullabilityKind, bool>;

const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
DiagNullabilityKind nullability);

inline DiagnosticBuilder DiagnosticsEngine::Report(SourceLocation Loc,
unsigned DiagID) {
assert(CurDiagID == std::numeric_limits<unsigned>::max() &&
"Multiple diagnostics in flight at once!");
CurDiagLoc = Loc;
CurDiagID = DiagID;
FlagValue.clear();
return DiagnosticBuilder(this);
}

inline DiagnosticBuilder DiagnosticsEngine::Report(unsigned DiagID) {
return Report(SourceLocation(), DiagID);
}


class Diagnostic {
const DiagnosticsEngine *DiagObj;
StringRef StoredDiagMessage;

public:
explicit Diagnostic(const DiagnosticsEngine *DO) : DiagObj(DO) {}
Diagnostic(const DiagnosticsEngine *DO, StringRef storedDiagMessage)
: DiagObj(DO), StoredDiagMessage(storedDiagMessage) {}

const DiagnosticsEngine *getDiags() const { return DiagObj; }
unsigned getID() const { return DiagObj->CurDiagID; }
const SourceLocation &getLocation() const { return DiagObj->CurDiagLoc; }
bool hasSourceManager() const { return DiagObj->hasSourceManager(); }
SourceManager &getSourceManager() const { return DiagObj->getSourceManager();}

unsigned getNumArgs() const { return DiagObj->NumDiagArgs; }

DiagnosticsEngine::ArgumentKind getArgKind(unsigned Idx) const {
assert(Idx < getNumArgs() && "Argument index out of range!");
return (DiagnosticsEngine::ArgumentKind)DiagObj->DiagArgumentsKind[Idx];
}

const std::string &getArgStdStr(unsigned Idx) const {
assert(getArgKind(Idx) == DiagnosticsEngine::ak_std_string &&
"invalid argument accessor!");
return DiagObj->DiagArgumentsStr[Idx];
}

const char *getArgCStr(unsigned Idx) const {
assert(getArgKind(Idx) == DiagnosticsEngine::ak_c_string &&
"invalid argument accessor!");
return reinterpret_cast<const char*>(DiagObj->DiagArgumentsVal[Idx]);
}

int getArgSInt(unsigned Idx) const {
assert(getArgKind(Idx) == DiagnosticsEngine::ak_sint &&
"invalid argument accessor!");
return (int)DiagObj->DiagArgumentsVal[Idx];
}

unsigned getArgUInt(unsigned Idx) const {
assert(getArgKind(Idx) == DiagnosticsEngine::ak_uint &&
"invalid argument accessor!");
return (unsigned)DiagObj->DiagArgumentsVal[Idx];
}

const IdentifierInfo *getArgIdentifier(unsigned Idx) const {
assert(getArgKind(Idx) == DiagnosticsEngine::ak_identifierinfo &&
"invalid argument accessor!");
return reinterpret_cast<IdentifierInfo*>(DiagObj->DiagArgumentsVal[Idx]);
}

intptr_t getRawArg(unsigned Idx) const {
assert(getArgKind(Idx) != DiagnosticsEngine::ak_std_string &&
"invalid argument accessor!");
return DiagObj->DiagArgumentsVal[Idx];
}

unsigned getNumRanges() const {
return DiagObj->DiagRanges.size();
}

const CharSourceRange &getRange(unsigned Idx) const {
assert(Idx < getNumRanges() && "Invalid diagnostic range index!");
return DiagObj->DiagRanges[Idx];
}

ArrayRef<CharSourceRange> getRanges() const {
return DiagObj->DiagRanges;
}

unsigned getNumFixItHints() const {
return DiagObj->DiagFixItHints.size();
}

const FixItHint &getFixItHint(unsigned Idx) const {
assert(Idx < getNumFixItHints() && "Invalid index!");
return DiagObj->DiagFixItHints[Idx];
}

ArrayRef<FixItHint> getFixItHints() const {
return DiagObj->DiagFixItHints;
}

void FormatDiagnostic(SmallVectorImpl<char> &OutStr) const;

void FormatDiagnostic(const char *DiagStr, const char *DiagEnd,
SmallVectorImpl<char> &OutStr) const;
};


class StoredDiagnostic {
unsigned ID;
DiagnosticsEngine::Level Level;
FullSourceLoc Loc;
std::string Message;
std::vector<CharSourceRange> Ranges;
std::vector<FixItHint> FixIts;

public:
StoredDiagnostic() = default;
StoredDiagnostic(DiagnosticsEngine::Level Level, const Diagnostic &Info);
StoredDiagnostic(DiagnosticsEngine::Level Level, unsigned ID,
StringRef Message);
StoredDiagnostic(DiagnosticsEngine::Level Level, unsigned ID,
StringRef Message, FullSourceLoc Loc,
ArrayRef<CharSourceRange> Ranges,
ArrayRef<FixItHint> Fixits);

explicit operator bool() const { return !Message.empty(); }

unsigned getID() const { return ID; }
DiagnosticsEngine::Level getLevel() const { return Level; }
const FullSourceLoc &getLocation() const { return Loc; }
StringRef getMessage() const { return Message; }

void setLocation(FullSourceLoc Loc) { this->Loc = Loc; }

using range_iterator = std::vector<CharSourceRange>::const_iterator;

range_iterator range_begin() const { return Ranges.begin(); }
range_iterator range_end() const { return Ranges.end(); }
unsigned range_size() const { return Ranges.size(); }

ArrayRef<CharSourceRange> getRanges() const {
return llvm::makeArrayRef(Ranges);
}

using fixit_iterator = std::vector<FixItHint>::const_iterator;

fixit_iterator fixit_begin() const { return FixIts.begin(); }
fixit_iterator fixit_end() const { return FixIts.end(); }
unsigned fixit_size() const { return FixIts.size(); }

ArrayRef<FixItHint> getFixIts() const {
return llvm::makeArrayRef(FixIts);
}
};

class DiagnosticConsumer {
protected:
unsigned NumWarnings = 0;       
unsigned NumErrors = 0;         

public:
DiagnosticConsumer() = default;
virtual ~DiagnosticConsumer();

unsigned getNumErrors() const { return NumErrors; }
unsigned getNumWarnings() const { return NumWarnings; }
virtual void clear() { NumWarnings = NumErrors = 0; }

virtual void BeginSourceFile(const LangOptions &LangOpts,
const Preprocessor *PP = nullptr) {}

virtual void EndSourceFile() {}

virtual void finish() {}

virtual bool IncludeInDiagnosticCounts() const;

virtual void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
const Diagnostic &Info);
};

class IgnoringDiagConsumer : public DiagnosticConsumer {
virtual void anchor();

void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
const Diagnostic &Info) override {
}
};

class ForwardingDiagnosticConsumer : public DiagnosticConsumer {
DiagnosticConsumer &Target;

public:
ForwardingDiagnosticConsumer(DiagnosticConsumer &Target) : Target(Target) {}
~ForwardingDiagnosticConsumer() override;

void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
const Diagnostic &Info) override;
void clear() override;

bool IncludeInDiagnosticCounts() const override;
};

struct TemplateDiffTypes {
intptr_t FromType;
intptr_t ToType;
unsigned PrintTree : 1;
unsigned PrintFromType : 1;
unsigned ElideType : 1;
unsigned ShowColors : 1;

unsigned TemplateDiffUsed : 1;
};

const char ToggleHighlight = 127;

void ProcessWarningOptions(DiagnosticsEngine &Diags,
const DiagnosticOptions &Opts,
bool ReportDiags = true);

} 

#endif 
