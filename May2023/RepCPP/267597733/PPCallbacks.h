
#ifndef LLVM_CLANG_LEX_PPCALLBACKS_H
#define LLVM_CLANG_LEX_PPCALLBACKS_H

#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/ModuleLoader.h"
#include "clang/Lex/Pragma.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
class Token;
class IdentifierInfo;
class MacroDefinition;
class MacroDirective;
class MacroArgs;

class PPCallbacks {
public:
virtual ~PPCallbacks();

enum FileChangeReason {
EnterFile, ExitFile, SystemHeaderPragma, RenameFile
};

virtual void FileChanged(SourceLocation Loc, FileChangeReason Reason,
SrcMgr::CharacteristicKind FileType,
FileID PrevFID = FileID()) {
}

virtual void FileSkipped(const FileEntry &SkippedFile,
const Token &FilenameTok,
SrcMgr::CharacteristicKind FileType) {
}

virtual bool FileNotFound(StringRef FileName,
SmallVectorImpl<char> &RecoveryPath) {
return false;
}

virtual void InclusionDirective(SourceLocation HashLoc,
const Token &IncludeTok,
StringRef FileName,
bool IsAngled,
CharSourceRange FilenameRange,
const FileEntry *File,
StringRef SearchPath,
StringRef RelativePath,
const Module *Imported,
SrcMgr::CharacteristicKind FileType) {
}

virtual void moduleImport(SourceLocation ImportLoc,
ModuleIdPath Path,
const Module *Imported) {
}

virtual void EndOfMainFile() {
}

virtual void Ident(SourceLocation Loc, StringRef str) {
}

virtual void PragmaDirective(SourceLocation Loc,
PragmaIntroducerKind Introducer) {
}

virtual void PragmaComment(SourceLocation Loc, const IdentifierInfo *Kind,
StringRef Str) {
}

virtual void PragmaDetectMismatch(SourceLocation Loc, StringRef Name,
StringRef Value) {
}

virtual void PragmaDebug(SourceLocation Loc, StringRef DebugType) {
}

enum PragmaMessageKind {
PMK_Message,

PMK_Warning,

PMK_Error
};

virtual void PragmaMessage(SourceLocation Loc, StringRef Namespace,
PragmaMessageKind Kind, StringRef Str) {
}

virtual void PragmaDiagnosticPush(SourceLocation Loc,
StringRef Namespace) {
}

virtual void PragmaDiagnosticPop(SourceLocation Loc,
StringRef Namespace) {
}

virtual void PragmaDiagnostic(SourceLocation Loc, StringRef Namespace,
diag::Severity mapping, StringRef Str) {}

virtual void PragmaOpenCLExtension(SourceLocation NameLoc,
const IdentifierInfo *Name,
SourceLocation StateLoc, unsigned State) {
}

virtual void PragmaWarning(SourceLocation Loc, StringRef WarningSpec,
ArrayRef<int> Ids) {
}

virtual void PragmaWarningPush(SourceLocation Loc, int Level) {
}

virtual void PragmaWarningPop(SourceLocation Loc) {
}

virtual void PragmaAssumeNonNullBegin(SourceLocation Loc) {}

virtual void PragmaAssumeNonNullEnd(SourceLocation Loc) {}

virtual void MacroExpands(const Token &MacroNameTok,
const MacroDefinition &MD, SourceRange Range,
const MacroArgs *Args) {}

virtual void MacroDefined(const Token &MacroNameTok,
const MacroDirective *MD) {
}

virtual void MacroUndefined(const Token &MacroNameTok,
const MacroDefinition &MD,
const MacroDirective *Undef) {
}

virtual void Defined(const Token &MacroNameTok, const MacroDefinition &MD,
SourceRange Range) {
}

virtual void SourceRangeSkipped(SourceRange Range, SourceLocation EndifLoc) {
}

enum ConditionValueKind {
CVK_NotEvaluated, CVK_False, CVK_True
};

virtual void If(SourceLocation Loc, SourceRange ConditionRange,
ConditionValueKind ConditionValue) {
}

virtual void Elif(SourceLocation Loc, SourceRange ConditionRange,
ConditionValueKind ConditionValue, SourceLocation IfLoc) {
}

virtual void Ifdef(SourceLocation Loc, const Token &MacroNameTok,
const MacroDefinition &MD) {
}

virtual void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
const MacroDefinition &MD) {
}

virtual void Else(SourceLocation Loc, SourceLocation IfLoc) {
}

virtual void Endif(SourceLocation Loc, SourceLocation IfLoc) {
}
};

class PPChainedCallbacks : public PPCallbacks {
virtual void anchor();
std::unique_ptr<PPCallbacks> First, Second;

public:
PPChainedCallbacks(std::unique_ptr<PPCallbacks> _First,
std::unique_ptr<PPCallbacks> _Second)
: First(std::move(_First)), Second(std::move(_Second)) {}

void FileChanged(SourceLocation Loc, FileChangeReason Reason,
SrcMgr::CharacteristicKind FileType,
FileID PrevFID) override {
First->FileChanged(Loc, Reason, FileType, PrevFID);
Second->FileChanged(Loc, Reason, FileType, PrevFID);
}

void FileSkipped(const FileEntry &SkippedFile,
const Token &FilenameTok,
SrcMgr::CharacteristicKind FileType) override {
First->FileSkipped(SkippedFile, FilenameTok, FileType);
Second->FileSkipped(SkippedFile, FilenameTok, FileType);
}

bool FileNotFound(StringRef FileName,
SmallVectorImpl<char> &RecoveryPath) override {
return First->FileNotFound(FileName, RecoveryPath) ||
Second->FileNotFound(FileName, RecoveryPath);
}

void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
StringRef FileName, bool IsAngled,
CharSourceRange FilenameRange, const FileEntry *File,
StringRef SearchPath, StringRef RelativePath,
const Module *Imported,
SrcMgr::CharacteristicKind FileType) override {
First->InclusionDirective(HashLoc, IncludeTok, FileName, IsAngled,
FilenameRange, File, SearchPath, RelativePath,
Imported, FileType);
Second->InclusionDirective(HashLoc, IncludeTok, FileName, IsAngled,
FilenameRange, File, SearchPath, RelativePath,
Imported, FileType);
}

void moduleImport(SourceLocation ImportLoc, ModuleIdPath Path,
const Module *Imported) override {
First->moduleImport(ImportLoc, Path, Imported);
Second->moduleImport(ImportLoc, Path, Imported);
}

void EndOfMainFile() override {
First->EndOfMainFile();
Second->EndOfMainFile();
}

void Ident(SourceLocation Loc, StringRef str) override {
First->Ident(Loc, str);
Second->Ident(Loc, str);
}

void PragmaDirective(SourceLocation Loc,
PragmaIntroducerKind Introducer) override {
First->PragmaDirective(Loc, Introducer);
Second->PragmaDirective(Loc, Introducer);
}

void PragmaComment(SourceLocation Loc, const IdentifierInfo *Kind,
StringRef Str) override {
First->PragmaComment(Loc, Kind, Str);
Second->PragmaComment(Loc, Kind, Str);
}

void PragmaDetectMismatch(SourceLocation Loc, StringRef Name,
StringRef Value) override {
First->PragmaDetectMismatch(Loc, Name, Value);
Second->PragmaDetectMismatch(Loc, Name, Value);
}

void PragmaDebug(SourceLocation Loc, StringRef DebugType) override {
First->PragmaDebug(Loc, DebugType);
Second->PragmaDebug(Loc, DebugType);
}

void PragmaMessage(SourceLocation Loc, StringRef Namespace,
PragmaMessageKind Kind, StringRef Str) override {
First->PragmaMessage(Loc, Namespace, Kind, Str);
Second->PragmaMessage(Loc, Namespace, Kind, Str);
}

void PragmaDiagnosticPush(SourceLocation Loc, StringRef Namespace) override {
First->PragmaDiagnosticPush(Loc, Namespace);
Second->PragmaDiagnosticPush(Loc, Namespace);
}

void PragmaDiagnosticPop(SourceLocation Loc, StringRef Namespace) override {
First->PragmaDiagnosticPop(Loc, Namespace);
Second->PragmaDiagnosticPop(Loc, Namespace);
}

void PragmaDiagnostic(SourceLocation Loc, StringRef Namespace,
diag::Severity mapping, StringRef Str) override {
First->PragmaDiagnostic(Loc, Namespace, mapping, Str);
Second->PragmaDiagnostic(Loc, Namespace, mapping, Str);
}

void PragmaOpenCLExtension(SourceLocation NameLoc, const IdentifierInfo *Name,
SourceLocation StateLoc, unsigned State) override {
First->PragmaOpenCLExtension(NameLoc, Name, StateLoc, State);
Second->PragmaOpenCLExtension(NameLoc, Name, StateLoc, State);
}

void PragmaWarning(SourceLocation Loc, StringRef WarningSpec,
ArrayRef<int> Ids) override {
First->PragmaWarning(Loc, WarningSpec, Ids);
Second->PragmaWarning(Loc, WarningSpec, Ids);
}

void PragmaWarningPush(SourceLocation Loc, int Level) override {
First->PragmaWarningPush(Loc, Level);
Second->PragmaWarningPush(Loc, Level);
}

void PragmaWarningPop(SourceLocation Loc) override {
First->PragmaWarningPop(Loc);
Second->PragmaWarningPop(Loc);
}

void PragmaAssumeNonNullBegin(SourceLocation Loc) override {
First->PragmaAssumeNonNullBegin(Loc);
Second->PragmaAssumeNonNullBegin(Loc);
}

void PragmaAssumeNonNullEnd(SourceLocation Loc) override {
First->PragmaAssumeNonNullEnd(Loc);
Second->PragmaAssumeNonNullEnd(Loc);
}

void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
SourceRange Range, const MacroArgs *Args) override {
First->MacroExpands(MacroNameTok, MD, Range, Args);
Second->MacroExpands(MacroNameTok, MD, Range, Args);
}

void MacroDefined(const Token &MacroNameTok,
const MacroDirective *MD) override {
First->MacroDefined(MacroNameTok, MD);
Second->MacroDefined(MacroNameTok, MD);
}

void MacroUndefined(const Token &MacroNameTok,
const MacroDefinition &MD,
const MacroDirective *Undef) override {
First->MacroUndefined(MacroNameTok, MD, Undef);
Second->MacroUndefined(MacroNameTok, MD, Undef);
}

void Defined(const Token &MacroNameTok, const MacroDefinition &MD,
SourceRange Range) override {
First->Defined(MacroNameTok, MD, Range);
Second->Defined(MacroNameTok, MD, Range);
}

void SourceRangeSkipped(SourceRange Range, SourceLocation EndifLoc) override {
First->SourceRangeSkipped(Range, EndifLoc);
Second->SourceRangeSkipped(Range, EndifLoc);
}

void If(SourceLocation Loc, SourceRange ConditionRange,
ConditionValueKind ConditionValue) override {
First->If(Loc, ConditionRange, ConditionValue);
Second->If(Loc, ConditionRange, ConditionValue);
}

void Elif(SourceLocation Loc, SourceRange ConditionRange,
ConditionValueKind ConditionValue, SourceLocation IfLoc) override {
First->Elif(Loc, ConditionRange, ConditionValue, IfLoc);
Second->Elif(Loc, ConditionRange, ConditionValue, IfLoc);
}

void Ifdef(SourceLocation Loc, const Token &MacroNameTok,
const MacroDefinition &MD) override {
First->Ifdef(Loc, MacroNameTok, MD);
Second->Ifdef(Loc, MacroNameTok, MD);
}

void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
const MacroDefinition &MD) override {
First->Ifndef(Loc, MacroNameTok, MD);
Second->Ifndef(Loc, MacroNameTok, MD);
}

void Else(SourceLocation Loc, SourceLocation IfLoc) override {
First->Else(Loc, IfLoc);
Second->Else(Loc, IfLoc);
}

void Endif(SourceLocation Loc, SourceLocation IfLoc) override {
First->Endif(Loc, IfLoc);
Second->Endif(Loc, IfLoc);
}
};

}  

#endif
