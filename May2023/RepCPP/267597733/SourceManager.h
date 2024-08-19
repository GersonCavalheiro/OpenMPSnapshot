
#ifndef LLVM_CLANG_BASIC_SOURCEMANAGER_H
#define LLVM_CLANG_BASIC_SOURCEMANAGER_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cassert>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace clang {

class ASTReader;
class ASTWriter;
class LineTableInfo;
class SourceManager;

namespace SrcMgr {

enum CharacteristicKind {
C_User, C_System, C_ExternCSystem, C_User_ModuleMap, C_System_ModuleMap
};

inline bool isSystem(CharacteristicKind CK) {
return CK != C_User && CK != C_User_ModuleMap;
}

inline bool isModuleMap(CharacteristicKind CK) {
return CK == C_User_ModuleMap || CK == C_System_ModuleMap;
}

class alignas(8) ContentCache {
enum CCFlags {
InvalidFlag = 0x01,

DoNotFreeFlag = 0x02
};

mutable llvm::PointerIntPair<llvm::MemoryBuffer *, 2> Buffer;

public:
const FileEntry *OrigEntry;

const FileEntry *ContentsEntry;

unsigned *SourceLineCache = nullptr;

unsigned NumLines = 0;

unsigned BufferOverridden : 1;

unsigned IsSystemFile : 1;

unsigned IsTransient : 1;

ContentCache(const FileEntry *Ent = nullptr) : ContentCache(Ent, Ent) {}

ContentCache(const FileEntry *Ent, const FileEntry *contentEnt)
: Buffer(nullptr, false), OrigEntry(Ent), ContentsEntry(contentEnt),
BufferOverridden(false), IsSystemFile(false), IsTransient(false) {}

ContentCache(const ContentCache &RHS)
: Buffer(nullptr, false), BufferOverridden(false), IsSystemFile(false),
IsTransient(false) {
OrigEntry = RHS.OrigEntry;
ContentsEntry = RHS.ContentsEntry;

assert(RHS.Buffer.getPointer() == nullptr &&
RHS.SourceLineCache == nullptr &&
"Passed ContentCache object cannot own a buffer.");

NumLines = RHS.NumLines;
}

ContentCache &operator=(const ContentCache& RHS) = delete;

~ContentCache();

llvm::MemoryBuffer *getBuffer(DiagnosticsEngine &Diag,
const SourceManager &SM,
SourceLocation Loc = SourceLocation(),
bool *Invalid = nullptr) const;

unsigned getSize() const;

unsigned getSizeBytesMapped() const;

llvm::MemoryBuffer::BufferKind getMemoryBufferKind() const;

llvm::MemoryBuffer *getRawBuffer() const { return Buffer.getPointer(); }

void replaceBuffer(llvm::MemoryBuffer *B, bool DoNotFree = false);

bool isBufferInvalid() const {
return Buffer.getInt() & InvalidFlag;
}

bool shouldFreeBuffer() const {
return (Buffer.getInt() & DoNotFreeFlag) == 0;
}
};

static_assert(alignof(ContentCache) >= 8,
"ContentCache must be 8-byte aligned.");

class FileInfo {
friend class clang::SourceManager;
friend class clang::ASTWriter;
friend class clang::ASTReader;

unsigned IncludeLoc;  

unsigned NumCreatedFIDs : 31;

unsigned HasLineDirectives : 1;

llvm::PointerIntPair<const ContentCache*, 3, CharacteristicKind>
ContentAndKind;

public:
static FileInfo get(SourceLocation IL, const ContentCache *Con,
CharacteristicKind FileCharacter) {
FileInfo X;
X.IncludeLoc = IL.getRawEncoding();
X.NumCreatedFIDs = 0;
X.HasLineDirectives = false;
X.ContentAndKind.setPointer(Con);
X.ContentAndKind.setInt(FileCharacter);
return X;
}

SourceLocation getIncludeLoc() const {
return SourceLocation::getFromRawEncoding(IncludeLoc);
}

const ContentCache *getContentCache() const {
return ContentAndKind.getPointer();
}

CharacteristicKind getFileCharacteristic() const {
return ContentAndKind.getInt();
}

bool hasLineDirectives() const { return HasLineDirectives; }

void setHasLineDirectives() {
HasLineDirectives = true;
}
};

class ExpansionInfo {

unsigned SpellingLoc;

unsigned ExpansionLocStart, ExpansionLocEnd;

bool ExpansionIsTokenRange;

public:
SourceLocation getSpellingLoc() const {
SourceLocation SpellLoc = SourceLocation::getFromRawEncoding(SpellingLoc);
return SpellLoc.isInvalid() ? getExpansionLocStart() : SpellLoc;
}

SourceLocation getExpansionLocStart() const {
return SourceLocation::getFromRawEncoding(ExpansionLocStart);
}

SourceLocation getExpansionLocEnd() const {
SourceLocation EndLoc =
SourceLocation::getFromRawEncoding(ExpansionLocEnd);
return EndLoc.isInvalid() ? getExpansionLocStart() : EndLoc;
}

bool isExpansionTokenRange() const {
return ExpansionIsTokenRange;
}

CharSourceRange getExpansionLocRange() const {
return CharSourceRange(
SourceRange(getExpansionLocStart(), getExpansionLocEnd()),
isExpansionTokenRange());
}

bool isMacroArgExpansion() const {
return getExpansionLocStart().isValid() &&
SourceLocation::getFromRawEncoding(ExpansionLocEnd).isInvalid();
}

bool isMacroBodyExpansion() const {
return getExpansionLocStart().isValid() &&
SourceLocation::getFromRawEncoding(ExpansionLocEnd).isValid();
}

bool isFunctionMacroExpansion() const {
return getExpansionLocStart().isValid() &&
getExpansionLocStart() != getExpansionLocEnd();
}

static ExpansionInfo create(SourceLocation SpellingLoc,
SourceLocation Start, SourceLocation End,
bool ExpansionIsTokenRange = true) {
ExpansionInfo X;
X.SpellingLoc = SpellingLoc.getRawEncoding();
X.ExpansionLocStart = Start.getRawEncoding();
X.ExpansionLocEnd = End.getRawEncoding();
X.ExpansionIsTokenRange = ExpansionIsTokenRange;
return X;
}

static ExpansionInfo createForMacroArg(SourceLocation SpellingLoc,
SourceLocation ExpansionLoc) {
return create(SpellingLoc, ExpansionLoc, SourceLocation());
}

static ExpansionInfo createForTokenSplit(SourceLocation SpellingLoc,
SourceLocation Start,
SourceLocation End) {
return create(SpellingLoc, Start, End, false);
}
};

class SLocEntry {
unsigned Offset : 31;
unsigned IsExpansion : 1;
union {
FileInfo File;
ExpansionInfo Expansion;
};

public:
SLocEntry() : Offset(), IsExpansion(), File() {}

unsigned getOffset() const { return Offset; }

bool isExpansion() const { return IsExpansion; }
bool isFile() const { return !isExpansion(); }

const FileInfo &getFile() const {
assert(isFile() && "Not a file SLocEntry!");
return File;
}

const ExpansionInfo &getExpansion() const {
assert(isExpansion() && "Not a macro expansion SLocEntry!");
return Expansion;
}

static SLocEntry get(unsigned Offset, const FileInfo &FI) {
assert(!(Offset & (1 << 31)) && "Offset is too large");
SLocEntry E;
E.Offset = Offset;
E.IsExpansion = false;
E.File = FI;
return E;
}

static SLocEntry get(unsigned Offset, const ExpansionInfo &Expansion) {
assert(!(Offset & (1 << 31)) && "Offset is too large");
SLocEntry E;
E.Offset = Offset;
E.IsExpansion = true;
E.Expansion = Expansion;
return E;
}
};

} 

class ExternalSLocEntrySource {
public:
virtual ~ExternalSLocEntrySource();

virtual bool ReadSLocEntry(int ID) = 0;

virtual std::pair<SourceLocation, StringRef> getModuleImportLoc(int ID) = 0;
};

class InBeforeInTUCacheEntry {
FileID LQueryFID, RQueryFID;

bool IsLQFIDBeforeRQFID;

FileID CommonFID;

unsigned LCommonOffset, RCommonOffset;

public:
bool isCacheValid(FileID LHS, FileID RHS) const {
return LQueryFID == LHS && RQueryFID == RHS;
}

bool getCachedResult(unsigned LOffset, unsigned ROffset) const {
if (LQueryFID != CommonFID) LOffset = LCommonOffset;
if (RQueryFID != CommonFID) ROffset = RCommonOffset;

if (LOffset == ROffset)
return IsLQFIDBeforeRQFID;

return LOffset < ROffset;
}

void setQueryFIDs(FileID LHS, FileID RHS, bool isLFIDBeforeRFID) {
assert(LHS != RHS);
LQueryFID = LHS;
RQueryFID = RHS;
IsLQFIDBeforeRQFID = isLFIDBeforeRFID;
}

void clear() {
LQueryFID = RQueryFID = FileID();
IsLQFIDBeforeRQFID = false;
}

void setCommonLoc(FileID commonFID, unsigned lCommonOffset,
unsigned rCommonOffset) {
CommonFID = commonFID;
LCommonOffset = lCommonOffset;
RCommonOffset = rCommonOffset;
}
};

using ModuleBuildStack = ArrayRef<std::pair<std::string, FullSourceLoc>>;

class SourceManager : public RefCountedBase<SourceManager> {
DiagnosticsEngine &Diag;

FileManager &FileMgr;

mutable llvm::BumpPtrAllocator ContentCacheAlloc;

llvm::DenseMap<const FileEntry*, SrcMgr::ContentCache*> FileInfos;

bool OverridenFilesKeepOriginalName = true;

bool UserFilesAreVolatile;

bool FilesAreTransient = false;

struct OverriddenFilesInfoTy {
llvm::DenseMap<const FileEntry *, const FileEntry *> OverriddenFiles;

llvm::DenseSet<const FileEntry *> OverriddenFilesWithBuffer;
};

std::unique_ptr<OverriddenFilesInfoTy> OverriddenFilesInfo;

OverriddenFilesInfoTy &getOverriddenFilesInfo() {
if (!OverriddenFilesInfo)
OverriddenFilesInfo.reset(new OverriddenFilesInfoTy);
return *OverriddenFilesInfo;
}

std::vector<SrcMgr::ContentCache*> MemBufferInfos;

SmallVector<SrcMgr::SLocEntry, 0> LocalSLocEntryTable;

mutable SmallVector<SrcMgr::SLocEntry, 0> LoadedSLocEntryTable;

unsigned NextLocalOffset;

unsigned CurrentLoadedOffset;

static const unsigned MaxLoadedOffset = 1U << 31U;

llvm::BitVector SLocEntryLoaded;

ExternalSLocEntrySource *ExternalSLocEntries = nullptr;

mutable FileID LastFileIDLookup;

LineTableInfo *LineTable = nullptr;

mutable FileID LastLineNoFileIDQuery;
mutable SrcMgr::ContentCache *LastLineNoContentCache;
mutable unsigned LastLineNoFilePos;
mutable unsigned LastLineNoResult;

FileID MainFileID;

FileID PreambleFileID;

mutable unsigned NumLinearScans = 0;
mutable unsigned NumBinaryProbes = 0;

mutable llvm::DenseMap<FileID, std::pair<FileID, unsigned>> IncludedLocMap;

using IsBeforeInTUCacheKey = std::pair<FileID, FileID>;

using InBeforeInTUCache =
llvm::DenseMap<IsBeforeInTUCacheKey, InBeforeInTUCacheEntry>;

mutable InBeforeInTUCache IBTUCache;
mutable InBeforeInTUCacheEntry IBTUCacheOverflow;

InBeforeInTUCacheEntry &getInBeforeInTUCache(FileID LFID, FileID RFID) const;

mutable std::unique_ptr<llvm::MemoryBuffer> FakeBufferForRecovery;

mutable std::unique_ptr<SrcMgr::ContentCache> FakeContentCacheForRecovery;

using MacroArgsMap = std::map<unsigned, SourceLocation>;

mutable llvm::DenseMap<FileID, std::unique_ptr<MacroArgsMap>>
MacroArgsCacheMap;

SmallVector<std::pair<std::string, FullSourceLoc>, 2> StoredModuleBuildStack;

public:
SourceManager(DiagnosticsEngine &Diag, FileManager &FileMgr,
bool UserFilesAreVolatile = false);
explicit SourceManager(const SourceManager &) = delete;
SourceManager &operator=(const SourceManager &) = delete;
~SourceManager();

void clearIDTables();

void initializeForReplay(const SourceManager &Old);

DiagnosticsEngine &getDiagnostics() const { return Diag; }

FileManager &getFileManager() const { return FileMgr; }

void setOverridenFilesKeepOriginalName(bool value) {
OverridenFilesKeepOriginalName = value;
}

bool userFilesAreVolatile() const { return UserFilesAreVolatile; }

ModuleBuildStack getModuleBuildStack() const {
return StoredModuleBuildStack;
}

void setModuleBuildStack(ModuleBuildStack stack) {
StoredModuleBuildStack.clear();
StoredModuleBuildStack.append(stack.begin(), stack.end());
}

void pushModuleBuildStack(StringRef moduleName, FullSourceLoc importLoc) {
StoredModuleBuildStack.push_back(std::make_pair(moduleName.str(),importLoc));
}


FileID getMainFileID() const { return MainFileID; }

void setMainFileID(FileID FID) {
MainFileID = FID;
}

void setPreambleFileID(FileID Preamble) {
assert(PreambleFileID.isInvalid() && "PreambleFileID already set!");
PreambleFileID = Preamble;
}

FileID getPreambleFileID() const { return PreambleFileID; }


FileID createFileID(const FileEntry *SourceFile, SourceLocation IncludePos,
SrcMgr::CharacteristicKind FileCharacter,
int LoadedID = 0, unsigned LoadedOffset = 0) {
const SrcMgr::ContentCache *IR =
getOrCreateContentCache(SourceFile, isSystem(FileCharacter));
assert(IR && "getOrCreateContentCache() cannot return NULL");
return createFileID(IR, IncludePos, FileCharacter, LoadedID, LoadedOffset);
}

FileID createFileID(std::unique_ptr<llvm::MemoryBuffer> Buffer,
SrcMgr::CharacteristicKind FileCharacter = SrcMgr::C_User,
int LoadedID = 0, unsigned LoadedOffset = 0,
SourceLocation IncludeLoc = SourceLocation()) {
return createFileID(
createMemBufferContentCache(Buffer.release(),  false),
IncludeLoc, FileCharacter, LoadedID, LoadedOffset);
}

enum UnownedTag { Unowned };

FileID createFileID(UnownedTag, llvm::MemoryBuffer *Buffer,
SrcMgr::CharacteristicKind FileCharacter = SrcMgr::C_User,
int LoadedID = 0, unsigned LoadedOffset = 0,
SourceLocation IncludeLoc = SourceLocation()) {
return createFileID(createMemBufferContentCache(Buffer, true),
IncludeLoc, FileCharacter, LoadedID, LoadedOffset);
}

FileID getOrCreateFileID(const FileEntry *SourceFile,
SrcMgr::CharacteristicKind FileCharacter) {
FileID ID = translateFile(SourceFile);
return ID.isValid() ? ID : createFileID(SourceFile, SourceLocation(),
FileCharacter);
}

SourceLocation createMacroArgExpansionLoc(SourceLocation Loc,
SourceLocation ExpansionLoc,
unsigned TokLength);

SourceLocation createExpansionLoc(SourceLocation Loc,
SourceLocation ExpansionLocStart,
SourceLocation ExpansionLocEnd,
unsigned TokLength,
bool ExpansionIsTokenRange = true,
int LoadedID = 0,
unsigned LoadedOffset = 0);

SourceLocation createTokenSplitLoc(SourceLocation SpellingLoc,
SourceLocation TokenStart,
SourceLocation TokenEnd);

llvm::MemoryBuffer *getMemoryBufferForFile(const FileEntry *File,
bool *Invalid = nullptr);

void overrideFileContents(const FileEntry *SourceFile,
llvm::MemoryBuffer *Buffer, bool DoNotFree);
void overrideFileContents(const FileEntry *SourceFile,
std::unique_ptr<llvm::MemoryBuffer> Buffer) {
overrideFileContents(SourceFile, Buffer.release(),  false);
}

void overrideFileContents(const FileEntry *SourceFile,
const FileEntry *NewFile);

bool isFileOverridden(const FileEntry *File) const {
if (OverriddenFilesInfo) {
if (OverriddenFilesInfo->OverriddenFilesWithBuffer.count(File))
return true;
if (OverriddenFilesInfo->OverriddenFiles.find(File) !=
OverriddenFilesInfo->OverriddenFiles.end())
return true;
}
return false;
}

void disableFileContentsOverride(const FileEntry *File);

void setFileIsTransient(const FileEntry *SourceFile);

void setAllFilesAreTransient(bool Transient) {
FilesAreTransient = Transient;
}


llvm::MemoryBuffer *getBuffer(FileID FID, SourceLocation Loc,
bool *Invalid = nullptr) const {
bool MyInvalid = false;
const SrcMgr::SLocEntry &Entry = getSLocEntry(FID, &MyInvalid);
if (MyInvalid || !Entry.isFile()) {
if (Invalid)
*Invalid = true;

return getFakeBufferForRecovery();
}

return Entry.getFile().getContentCache()->getBuffer(Diag, *this, Loc,
Invalid);
}

llvm::MemoryBuffer *getBuffer(FileID FID, bool *Invalid = nullptr) const {
bool MyInvalid = false;
const SrcMgr::SLocEntry &Entry = getSLocEntry(FID, &MyInvalid);
if (MyInvalid || !Entry.isFile()) {
if (Invalid)
*Invalid = true;

return getFakeBufferForRecovery();
}

return Entry.getFile().getContentCache()->getBuffer(Diag, *this,
SourceLocation(),
Invalid);
}

const FileEntry *getFileEntryForID(FileID FID) const {
bool MyInvalid = false;
const SrcMgr::SLocEntry &Entry = getSLocEntry(FID, &MyInvalid);
if (MyInvalid || !Entry.isFile())
return nullptr;

const SrcMgr::ContentCache *Content = Entry.getFile().getContentCache();
if (!Content)
return nullptr;
return Content->OrigEntry;
}

const FileEntry *getFileEntryForSLocEntry(const SrcMgr::SLocEntry &sloc) const
{
const SrcMgr::ContentCache *Content = sloc.getFile().getContentCache();
if (!Content)
return nullptr;
return Content->OrigEntry;
}

StringRef getBufferData(FileID FID, bool *Invalid = nullptr) const;

unsigned getNumCreatedFIDsForFileID(FileID FID) const {
bool Invalid = false;
const SrcMgr::SLocEntry &Entry = getSLocEntry(FID, &Invalid);
if (Invalid || !Entry.isFile())
return 0;

return Entry.getFile().NumCreatedFIDs;
}

void setNumCreatedFIDsForFileID(FileID FID, unsigned NumFIDs) const {
bool Invalid = false;
const SrcMgr::SLocEntry &Entry = getSLocEntry(FID, &Invalid);
if (Invalid || !Entry.isFile())
return;

assert(Entry.getFile().NumCreatedFIDs == 0 && "Already set!");
const_cast<SrcMgr::FileInfo &>(Entry.getFile()).NumCreatedFIDs = NumFIDs;
}


FileID getFileID(SourceLocation SpellingLoc) const {
unsigned SLocOffset = SpellingLoc.getOffset();

if (isOffsetInFileID(LastFileIDLookup, SLocOffset))
return LastFileIDLookup;

return getFileIDSlow(SLocOffset);
}

StringRef getFilename(SourceLocation SpellingLoc) const {
if (const FileEntry *F = getFileEntryForID(getFileID(SpellingLoc)))
return F->getName();
return StringRef();
}

SourceLocation getLocForStartOfFile(FileID FID) const {
bool Invalid = false;
const SrcMgr::SLocEntry &Entry = getSLocEntry(FID, &Invalid);
if (Invalid || !Entry.isFile())
return SourceLocation();

unsigned FileOffset = Entry.getOffset();
return SourceLocation::getFileLoc(FileOffset);
}

SourceLocation getLocForEndOfFile(FileID FID) const {
bool Invalid = false;
const SrcMgr::SLocEntry &Entry = getSLocEntry(FID, &Invalid);
if (Invalid || !Entry.isFile())
return SourceLocation();

unsigned FileOffset = Entry.getOffset();
return SourceLocation::getFileLoc(FileOffset + getFileIDSize(FID));
}

SourceLocation getIncludeLoc(FileID FID) const {
bool Invalid = false;
const SrcMgr::SLocEntry &Entry = getSLocEntry(FID, &Invalid);
if (Invalid || !Entry.isFile())
return SourceLocation();

return Entry.getFile().getIncludeLoc();
}

std::pair<SourceLocation, StringRef>
getModuleImportLoc(SourceLocation Loc) const {
FileID FID = getFileID(Loc);

if (FID.ID >= -1)
return std::make_pair(SourceLocation(), "");

return ExternalSLocEntries->getModuleImportLoc(FID.ID);
}

SourceLocation getExpansionLoc(SourceLocation Loc) const {
if (Loc.isFileID()) return Loc;
return getExpansionLocSlowCase(Loc);
}

SourceLocation getFileLoc(SourceLocation Loc) const {
if (Loc.isFileID()) return Loc;
return getFileLocSlowCase(Loc);
}

CharSourceRange getImmediateExpansionRange(SourceLocation Loc) const;

CharSourceRange getExpansionRange(SourceLocation Loc) const;

CharSourceRange getExpansionRange(SourceRange Range) const {
SourceLocation Begin = getExpansionRange(Range.getBegin()).getBegin();
CharSourceRange End = getExpansionRange(Range.getEnd());
return CharSourceRange(SourceRange(Begin, End.getEnd()),
End.isTokenRange());
}

CharSourceRange getExpansionRange(CharSourceRange Range) const {
CharSourceRange Expansion = getExpansionRange(Range.getAsRange());
if (Expansion.getEnd() == Range.getEnd())
Expansion.setTokenRange(Range.isTokenRange());
return Expansion;
}

SourceLocation getSpellingLoc(SourceLocation Loc) const {
if (Loc.isFileID()) return Loc;
return getSpellingLocSlowCase(Loc);
}

SourceLocation getImmediateSpellingLoc(SourceLocation Loc) const;

SourceLocation getComposedLoc(FileID FID, unsigned Offset) const {
bool Invalid = false;
const SrcMgr::SLocEntry &Entry = getSLocEntry(FID, &Invalid);
if (Invalid)
return SourceLocation();

unsigned GlobalOffset = Entry.getOffset() + Offset;
return Entry.isFile() ? SourceLocation::getFileLoc(GlobalOffset)
: SourceLocation::getMacroLoc(GlobalOffset);
}

std::pair<FileID, unsigned> getDecomposedLoc(SourceLocation Loc) const {
FileID FID = getFileID(Loc);
bool Invalid = false;
const SrcMgr::SLocEntry &E = getSLocEntry(FID, &Invalid);
if (Invalid)
return std::make_pair(FileID(), 0);
return std::make_pair(FID, Loc.getOffset()-E.getOffset());
}

std::pair<FileID, unsigned>
getDecomposedExpansionLoc(SourceLocation Loc) const {
FileID FID = getFileID(Loc);
bool Invalid = false;
const SrcMgr::SLocEntry *E = &getSLocEntry(FID, &Invalid);
if (Invalid)
return std::make_pair(FileID(), 0);

unsigned Offset = Loc.getOffset()-E->getOffset();
if (Loc.isFileID())
return std::make_pair(FID, Offset);

return getDecomposedExpansionLocSlowCase(E);
}

std::pair<FileID, unsigned>
getDecomposedSpellingLoc(SourceLocation Loc) const {
FileID FID = getFileID(Loc);
bool Invalid = false;
const SrcMgr::SLocEntry *E = &getSLocEntry(FID, &Invalid);
if (Invalid)
return std::make_pair(FileID(), 0);

unsigned Offset = Loc.getOffset()-E->getOffset();
if (Loc.isFileID())
return std::make_pair(FID, Offset);
return getDecomposedSpellingLocSlowCase(E, Offset);
}

std::pair<FileID, unsigned> getDecomposedIncludedLoc(FileID FID) const;

unsigned getFileOffset(SourceLocation SpellingLoc) const {
return getDecomposedLoc(SpellingLoc).second;
}

bool isMacroArgExpansion(SourceLocation Loc,
SourceLocation *StartLoc = nullptr) const;

bool isMacroBodyExpansion(SourceLocation Loc) const;

bool isAtStartOfImmediateMacroExpansion(SourceLocation Loc,
SourceLocation *MacroBegin = nullptr) const;

bool
isAtEndOfImmediateMacroExpansion(SourceLocation Loc,
SourceLocation *MacroEnd = nullptr) const;

bool isInSLocAddrSpace(SourceLocation Loc,
SourceLocation Start, unsigned Length,
unsigned *RelativeOffset = nullptr) const {
assert(((Start.getOffset() < NextLocalOffset &&
Start.getOffset()+Length <= NextLocalOffset) ||
(Start.getOffset() >= CurrentLoadedOffset &&
Start.getOffset()+Length < MaxLoadedOffset)) &&
"Chunk is not valid SLoc address space");
unsigned LocOffs = Loc.getOffset();
unsigned BeginOffs = Start.getOffset();
unsigned EndOffs = BeginOffs + Length;
if (LocOffs >= BeginOffs && LocOffs < EndOffs) {
if (RelativeOffset)
*RelativeOffset = LocOffs - BeginOffs;
return true;
}

return false;
}

bool isInSameSLocAddrSpace(SourceLocation LHS, SourceLocation RHS,
int *RelativeOffset) const {
unsigned LHSOffs = LHS.getOffset(), RHSOffs = RHS.getOffset();
bool LHSLoaded = LHSOffs >= CurrentLoadedOffset;
bool RHSLoaded = RHSOffs >= CurrentLoadedOffset;

if (LHSLoaded == RHSLoaded) {
if (RelativeOffset)
*RelativeOffset = RHSOffs - LHSOffs;
return true;
}

return false;
}


const char *getCharacterData(SourceLocation SL,
bool *Invalid = nullptr) const;

unsigned getColumnNumber(FileID FID, unsigned FilePos,
bool *Invalid = nullptr) const;
unsigned getSpellingColumnNumber(SourceLocation Loc,
bool *Invalid = nullptr) const;
unsigned getExpansionColumnNumber(SourceLocation Loc,
bool *Invalid = nullptr) const;
unsigned getPresumedColumnNumber(SourceLocation Loc,
bool *Invalid = nullptr) const;

unsigned getLineNumber(FileID FID, unsigned FilePos, bool *Invalid = nullptr) const;
unsigned getSpellingLineNumber(SourceLocation Loc, bool *Invalid = nullptr) const;
unsigned getExpansionLineNumber(SourceLocation Loc, bool *Invalid = nullptr) const;
unsigned getPresumedLineNumber(SourceLocation Loc, bool *Invalid = nullptr) const;

StringRef getBufferName(SourceLocation Loc, bool *Invalid = nullptr) const;

SrcMgr::CharacteristicKind getFileCharacteristic(SourceLocation Loc) const;

PresumedLoc getPresumedLoc(SourceLocation Loc,
bool UseLineDirectives = true) const;

bool isInMainFile(SourceLocation Loc) const;

bool isWrittenInSameFile(SourceLocation Loc1, SourceLocation Loc2) const {
return getFileID(Loc1) == getFileID(Loc2);
}

bool isWrittenInMainFile(SourceLocation Loc) const {
return getFileID(Loc) == getMainFileID();
}

bool isInSystemHeader(SourceLocation Loc) const {
return isSystem(getFileCharacteristic(Loc));
}

bool isInExternCSystemHeader(SourceLocation Loc) const {
return getFileCharacteristic(Loc) == SrcMgr::C_ExternCSystem;
}

bool isInSystemMacro(SourceLocation loc) const {
return loc.isMacroID() && isInSystemHeader(getSpellingLoc(loc));
}

unsigned getFileIDSize(FileID FID) const;

bool isInFileID(SourceLocation Loc, FileID FID,
unsigned *RelativeOffset = nullptr) const {
unsigned Offs = Loc.getOffset();
if (isOffsetInFileID(FID, Offs)) {
if (RelativeOffset)
*RelativeOffset = Offs - getSLocEntry(FID).getOffset();
return true;
}

return false;
}


unsigned getLineTableFilenameID(StringRef Str);

void AddLineNote(SourceLocation Loc, unsigned LineNo, int FilenameID,
bool IsFileEntry, bool IsFileExit,
SrcMgr::CharacteristicKind FileKind);

bool hasLineTable() const { return LineTable != nullptr; }

LineTableInfo &getLineTable();


size_t getContentCacheSize() const {
return ContentCacheAlloc.getTotalMemory();
}

struct MemoryBufferSizes {
const size_t malloc_bytes;
const size_t mmap_bytes;

MemoryBufferSizes(size_t malloc_bytes, size_t mmap_bytes)
: malloc_bytes(malloc_bytes), mmap_bytes(mmap_bytes) {}
};

MemoryBufferSizes getMemoryBufferSizes() const;

size_t getDataStructureSizes() const;


SourceLocation translateFileLineCol(const FileEntry *SourceFile,
unsigned Line, unsigned Col) const;

FileID translateFile(const FileEntry *SourceFile) const;

SourceLocation translateLineCol(FileID FID,
unsigned Line, unsigned Col) const;

SourceLocation getMacroArgExpandedLocation(SourceLocation Loc) const;

bool isBeforeInTranslationUnit(SourceLocation LHS, SourceLocation RHS) const;

std::pair<bool, bool>
isInTheSameTranslationUnit(std::pair<FileID, unsigned> &LOffs,
std::pair<FileID, unsigned> &ROffs) const;

bool isBeforeInSLocAddrSpace(SourceLocation LHS, SourceLocation RHS) const {
return isBeforeInSLocAddrSpace(LHS, RHS.getOffset());
}

bool isBeforeInSLocAddrSpace(SourceLocation LHS, unsigned RHS) const {
unsigned LHSOffset = LHS.getOffset();
bool LHSLoaded = LHSOffset >= CurrentLoadedOffset;
bool RHSLoaded = RHS >= CurrentLoadedOffset;
if (LHSLoaded == RHSLoaded)
return LHSOffset < RHS;

return LHSLoaded;
}

bool isPointWithin(SourceLocation Location, SourceLocation Start,
SourceLocation End) const {
return Location == Start || Location == End ||
(isBeforeInTranslationUnit(Start, Location) &&
isBeforeInTranslationUnit(Location, End));
}

using fileinfo_iterator =
llvm::DenseMap<const FileEntry*, SrcMgr::ContentCache*>::const_iterator;

fileinfo_iterator fileinfo_begin() const { return FileInfos.begin(); }
fileinfo_iterator fileinfo_end() const { return FileInfos.end(); }
bool hasFileInfo(const FileEntry *File) const {
return FileInfos.find(File) != FileInfos.end();
}

void PrintStats() const;

void dump() const;

unsigned local_sloc_entry_size() const { return LocalSLocEntryTable.size(); }

const SrcMgr::SLocEntry &getLocalSLocEntry(unsigned Index,
bool *Invalid = nullptr) const {
assert(Index < LocalSLocEntryTable.size() && "Invalid index");
return LocalSLocEntryTable[Index];
}

unsigned loaded_sloc_entry_size() const { return LoadedSLocEntryTable.size();}

const SrcMgr::SLocEntry &getLoadedSLocEntry(unsigned Index,
bool *Invalid = nullptr) const {
assert(Index < LoadedSLocEntryTable.size() && "Invalid index");
if (SLocEntryLoaded[Index])
return LoadedSLocEntryTable[Index];
return loadSLocEntry(Index, Invalid);
}

const SrcMgr::SLocEntry &getSLocEntry(FileID FID,
bool *Invalid = nullptr) const {
if (FID.ID == 0 || FID.ID == -1) {
if (Invalid) *Invalid = true;
return LocalSLocEntryTable[0];
}
return getSLocEntryByID(FID.ID, Invalid);
}

unsigned getNextLocalOffset() const { return NextLocalOffset; }

void setExternalSLocEntrySource(ExternalSLocEntrySource *Source) {
assert(LoadedSLocEntryTable.empty() &&
"Invalidating existing loaded entries");
ExternalSLocEntries = Source;
}

std::pair<int, unsigned>
AllocateLoadedSLocEntries(unsigned NumSLocEntries, unsigned TotalSize);

bool isLoadedSourceLocation(SourceLocation Loc) const {
return Loc.getOffset() >= CurrentLoadedOffset;
}

bool isLocalSourceLocation(SourceLocation Loc) const {
return Loc.getOffset() < NextLocalOffset;
}

bool isLoadedFileID(FileID FID) const {
assert(FID.ID != -1 && "Using FileID sentinel value");
return FID.ID < 0;
}

bool isLocalFileID(FileID FID) const {
return !isLoadedFileID(FID);
}

SourceLocation getImmediateMacroCallerLoc(SourceLocation Loc) const {
if (!Loc.isMacroID()) return Loc;

if (isMacroArgExpansion(Loc))
return getImmediateSpellingLoc(Loc);

return getImmediateExpansionRange(Loc).getBegin();
}

SourceLocation getTopMacroCallerLoc(SourceLocation Loc) const;

private:
friend class ASTReader;
friend class ASTWriter;

llvm::MemoryBuffer *getFakeBufferForRecovery() const;
const SrcMgr::ContentCache *getFakeContentCacheForRecovery() const;

const SrcMgr::SLocEntry &loadSLocEntry(unsigned Index, bool *Invalid) const;

const SrcMgr::SLocEntry &getSLocEntryByID(int ID,
bool *Invalid = nullptr) const {
assert(ID != -1 && "Using FileID sentinel value");
if (ID < 0)
return getLoadedSLocEntryByID(ID, Invalid);
return getLocalSLocEntry(static_cast<unsigned>(ID), Invalid);
}

const SrcMgr::SLocEntry &
getLoadedSLocEntryByID(int ID, bool *Invalid = nullptr) const {
return getLoadedSLocEntry(static_cast<unsigned>(-ID - 2), Invalid);
}

SourceLocation createExpansionLocImpl(const SrcMgr::ExpansionInfo &Expansion,
unsigned TokLength,
int LoadedID = 0,
unsigned LoadedOffset = 0);

inline bool isOffsetInFileID(FileID FID, unsigned SLocOffset) const {
const SrcMgr::SLocEntry &Entry = getSLocEntry(FID);
if (SLocOffset < Entry.getOffset()) return false;

if (FID.ID == -2)
return true;

if (FID.ID+1 == static_cast<int>(LocalSLocEntryTable.size()))
return SLocOffset < NextLocalOffset;

return SLocOffset < getSLocEntryByID(FID.ID+1).getOffset();
}

FileID getPreviousFileID(FileID FID) const;

FileID getNextFileID(FileID FID) const;

FileID createFileID(const SrcMgr::ContentCache* File,
SourceLocation IncludePos,
SrcMgr::CharacteristicKind DirCharacter,
int LoadedID, unsigned LoadedOffset);

const SrcMgr::ContentCache *
getOrCreateContentCache(const FileEntry *SourceFile,
bool isSystemFile = false);

const SrcMgr::ContentCache *
createMemBufferContentCache(llvm::MemoryBuffer *Buf, bool DoNotFree);

FileID getFileIDSlow(unsigned SLocOffset) const;
FileID getFileIDLocal(unsigned SLocOffset) const;
FileID getFileIDLoaded(unsigned SLocOffset) const;

SourceLocation getExpansionLocSlowCase(SourceLocation Loc) const;
SourceLocation getSpellingLocSlowCase(SourceLocation Loc) const;
SourceLocation getFileLocSlowCase(SourceLocation Loc) const;

std::pair<FileID, unsigned>
getDecomposedExpansionLocSlowCase(const SrcMgr::SLocEntry *E) const;
std::pair<FileID, unsigned>
getDecomposedSpellingLocSlowCase(const SrcMgr::SLocEntry *E,
unsigned Offset) const;
void computeMacroArgsCache(MacroArgsMap &MacroArgsCache, FileID FID) const;
void associateFileChunkWithMacroArgExp(MacroArgsMap &MacroArgsCache,
FileID FID,
SourceLocation SpellLoc,
SourceLocation ExpansionLoc,
unsigned ExpansionLength) const;
};

template<typename T>
class BeforeThanCompare;

template<>
class BeforeThanCompare<SourceLocation> {
SourceManager &SM;

public:
explicit BeforeThanCompare(SourceManager &SM) : SM(SM) {}

bool operator()(SourceLocation LHS, SourceLocation RHS) const {
return SM.isBeforeInTranslationUnit(LHS, RHS);
}
};

template<>
class BeforeThanCompare<SourceRange> {
SourceManager &SM;

public:
explicit BeforeThanCompare(SourceManager &SM) : SM(SM) {}

bool operator()(SourceRange LHS, SourceRange RHS) const {
return SM.isBeforeInTranslationUnit(LHS.getBegin(), RHS.getBegin());
}
};

class SourceManagerForFile {
public:
SourceManagerForFile(StringRef FileName, StringRef Content);

SourceManager &get() {
assert(SourceMgr);
return *SourceMgr;
}

private:
std::unique_ptr<FileManager> FileMgr;
std::unique_ptr<DiagnosticsEngine> Diagnostics;
std::unique_ptr<SourceManager> SourceMgr;
};

} 

#endif 
