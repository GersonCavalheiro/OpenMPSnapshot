
#ifndef LLVM_CLANG_LEX_HEADERSEARCH_H
#define LLVM_CLANG_LEX_HEADERSEARCH_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/DirectoryLookup.h"
#include "clang/Lex/ModuleMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace clang {

class DiagnosticsEngine;
class DirectoryEntry;
class ExternalPreprocessorSource;
class FileEntry;
class FileManager;
class HeaderMap;
class HeaderSearchOptions;
class IdentifierInfo;
class LangOptions;
class Module;
class Preprocessor;
class TargetInfo;

struct HeaderFileInfo {
unsigned isImport : 1;

unsigned isPragmaOnce : 1;

unsigned DirInfo : 3;

unsigned External : 1;

unsigned isModuleHeader : 1;

unsigned isCompilingModuleHeader : 1;

unsigned Resolved : 1;

unsigned IndexHeaderMapHeader : 1;

unsigned IsValid : 1;

unsigned short NumIncludes = 0;

unsigned ControllingMacroID = 0;

const IdentifierInfo *ControllingMacro = nullptr;

StringRef Framework;

HeaderFileInfo()
: isImport(false), isPragmaOnce(false), DirInfo(SrcMgr::C_User),
External(false), isModuleHeader(false), isCompilingModuleHeader(false),
Resolved(false), IndexHeaderMapHeader(false), IsValid(false)  {}

const IdentifierInfo *
getControllingMacro(ExternalPreprocessorSource *External);

bool isNonDefault() const {
return isImport || isPragmaOnce || NumIncludes || ControllingMacro ||
ControllingMacroID;
}
};

class ExternalHeaderFileInfoSource {
public:
virtual ~ExternalHeaderFileInfoSource();

virtual HeaderFileInfo GetHeaderFileInfo(const FileEntry *FE) = 0;
};

class HeaderSearch {
friend class DirectoryLookup;

struct FrameworkCacheEntry {
const DirectoryEntry *Directory;

bool IsUserSpecifiedSystemFramework;
};

std::shared_ptr<HeaderSearchOptions> HSOpts;

DiagnosticsEngine &Diags;
FileManager &FileMgr;

std::vector<DirectoryLookup> SearchDirs;
unsigned AngledDirIdx = 0;
unsigned SystemDirIdx = 0;
bool NoCurDirSearch = false;

std::vector<std::pair<std::string, bool>> SystemHeaderPrefixes;

std::string ModuleCachePath;

mutable std::vector<HeaderFileInfo> FileInfo;

struct LookupFileCacheInfo {
unsigned StartIdx = 0;

unsigned HitIdx = 0;

const char *MappedName = nullptr;

LookupFileCacheInfo() = default;

void reset(unsigned StartIdx) {
this->StartIdx = StartIdx;
this->MappedName = nullptr;
}
};
llvm::StringMap<LookupFileCacheInfo, llvm::BumpPtrAllocator> LookupFileCache;

llvm::StringMap<FrameworkCacheEntry, llvm::BumpPtrAllocator> FrameworkMap;

using IncludeAliasMap =
llvm::StringMap<std::string, llvm::BumpPtrAllocator>;
std::unique_ptr<IncludeAliasMap> IncludeAliases;

std::vector<std::pair<const FileEntry *, const HeaderMap *>> HeaderMaps;

mutable ModuleMap ModMap;

llvm::DenseMap<const DirectoryEntry *, bool> DirectoryHasModuleMap;

llvm::DenseMap<const FileEntry *, bool> LoadedModuleMaps;

llvm::StringSet<llvm::BumpPtrAllocator> FrameworkNames;

ExternalPreprocessorSource *ExternalLookup = nullptr;

ExternalHeaderFileInfoSource *ExternalSource = nullptr;

unsigned NumIncluded = 0;
unsigned NumMultiIncludeFileOptzn = 0;
unsigned NumFrameworkLookups = 0;
unsigned NumSubFrameworkLookups = 0;

public:
HeaderSearch(std::shared_ptr<HeaderSearchOptions> HSOpts,
SourceManager &SourceMgr, DiagnosticsEngine &Diags,
const LangOptions &LangOpts, const TargetInfo *Target);
HeaderSearch(const HeaderSearch &) = delete;
HeaderSearch &operator=(const HeaderSearch &) = delete;
~HeaderSearch();

HeaderSearchOptions &getHeaderSearchOpts() const { return *HSOpts; }

FileManager &getFileMgr() const { return FileMgr; }

DiagnosticsEngine &getDiags() const { return Diags; }

void SetSearchPaths(const std::vector<DirectoryLookup> &dirs,
unsigned angledDirIdx, unsigned systemDirIdx,
bool noCurDirSearch) {
assert(angledDirIdx <= systemDirIdx && systemDirIdx <= dirs.size() &&
"Directory indices are unordered");
SearchDirs = dirs;
AngledDirIdx = angledDirIdx;
SystemDirIdx = systemDirIdx;
NoCurDirSearch = noCurDirSearch;
}

void AddSearchPath(const DirectoryLookup &dir, bool isAngled) {
unsigned idx = isAngled ? SystemDirIdx : AngledDirIdx;
SearchDirs.insert(SearchDirs.begin() + idx, dir);
if (!isAngled)
AngledDirIdx++;
SystemDirIdx++;
}

void SetSystemHeaderPrefixes(ArrayRef<std::pair<std::string, bool>> P) {
SystemHeaderPrefixes.assign(P.begin(), P.end());
}

bool HasIncludeAliasMap() const { return (bool)IncludeAliases; }

void AddIncludeAlias(StringRef Source, StringRef Dest) {
if (!IncludeAliases)
IncludeAliases.reset(new IncludeAliasMap);
(*IncludeAliases)[Source] = Dest;
}

StringRef MapHeaderToIncludeAlias(StringRef Source) {
assert(IncludeAliases && "Trying to map headers when there's no map");

IncludeAliasMap::const_iterator Iter = IncludeAliases->find(Source);
if (Iter != IncludeAliases->end())
return Iter->second;
return {};
}

void setModuleCachePath(StringRef CachePath) {
ModuleCachePath = CachePath;
}

StringRef getModuleCachePath() const { return ModuleCachePath; }

void setDirectoryHasModuleMap(const DirectoryEntry* Dir) {
DirectoryHasModuleMap[Dir] = true;
}

void ClearFileInfo() {
FileInfo.clear();
}

void SetExternalLookup(ExternalPreprocessorSource *EPS) {
ExternalLookup = EPS;
}

ExternalPreprocessorSource *getExternalLookup() const {
return ExternalLookup;
}

void SetExternalSource(ExternalHeaderFileInfoSource *ES) {
ExternalSource = ES;
}

void setTarget(const TargetInfo &Target);

const FileEntry *LookupFile(
StringRef Filename, SourceLocation IncludeLoc, bool isAngled,
const DirectoryLookup *FromDir, const DirectoryLookup *&CurDir,
ArrayRef<std::pair<const FileEntry *, const DirectoryEntry *>> Includers,
SmallVectorImpl<char> *SearchPath, SmallVectorImpl<char> *RelativePath,
Module *RequestingModule, ModuleMap::KnownHeader *SuggestedModule,
bool *IsMapped, bool SkipCache = false, bool BuildSystemModule = false);

const FileEntry *LookupSubframeworkHeader(
StringRef Filename, const FileEntry *RelativeFileEnt,
SmallVectorImpl<char> *SearchPath, SmallVectorImpl<char> *RelativePath,
Module *RequestingModule, ModuleMap::KnownHeader *SuggestedModule);

FrameworkCacheEntry &LookupFrameworkCache(StringRef FWName) {
return FrameworkMap[FWName];
}

bool ShouldEnterIncludeFile(Preprocessor &PP, const FileEntry *File,
bool isImport, bool ModulesEnabled,
Module *CorrespondingModule);

SrcMgr::CharacteristicKind getFileDirFlavor(const FileEntry *File) {
return (SrcMgr::CharacteristicKind)getFileInfo(File).DirInfo;
}

void MarkFileIncludeOnce(const FileEntry *File) {
HeaderFileInfo &FI = getFileInfo(File);
FI.isImport = true;
FI.isPragmaOnce = true;
}

void MarkFileSystemHeader(const FileEntry *File) {
getFileInfo(File).DirInfo = SrcMgr::C_System;
}

void MarkFileModuleHeader(const FileEntry *File,
ModuleMap::ModuleHeaderRole Role,
bool IsCompiledModuleHeader);

void IncrementIncludeCount(const FileEntry *File) {
++getFileInfo(File).NumIncludes;
}

void SetFileControllingMacro(const FileEntry *File,
const IdentifierInfo *ControllingMacro) {
getFileInfo(File).ControllingMacro = ControllingMacro;
}

bool FirstTimeLexingFile(const FileEntry *File) {
return getFileInfo(File).NumIncludes == 1;
}

bool isFileMultipleIncludeGuarded(const FileEntry *File);

const HeaderMap *CreateHeaderMap(const FileEntry *FE);

void getHeaderMapFileNames(SmallVectorImpl<std::string> &Names) const;

std::string getCachedModuleFileName(Module *Module);

std::string getPrebuiltModuleFileName(StringRef ModuleName,
bool FileMapOnly = false);

std::string getCachedModuleFileName(StringRef ModuleName,
StringRef ModuleMapPath);

Module *lookupModule(StringRef ModuleName, bool AllowSearch = true,
bool AllowExtraModuleMapSearch = false);

const FileEntry *lookupModuleMapFile(const DirectoryEntry *Dir,
bool IsFramework);

void IncrementFrameworkLookupCount() { ++NumFrameworkLookups; }

bool hasModuleMap(StringRef Filename, const DirectoryEntry *Root,
bool IsSystem);

ModuleMap::KnownHeader findModuleForHeader(const FileEntry *File,
bool AllowTextual = false) const;

bool loadModuleMapFile(const FileEntry *File, bool IsSystem,
FileID ID = FileID(), unsigned *Offset = nullptr,
StringRef OriginalModuleMapFile = StringRef());

void collectAllModules(SmallVectorImpl<Module *> &Modules);

void loadTopLevelSystemModules();

private:
Module *lookupModule(StringRef ModuleName, StringRef SearchName,
bool AllowExtraModuleMapSearch = false);

Module *loadFrameworkModule(StringRef Name,
const DirectoryEntry *Dir,
bool IsSystem);

void loadSubdirectoryModuleMaps(DirectoryLookup &SearchDir);

bool findUsableModuleForHeader(const FileEntry *File,
const DirectoryEntry *Root,
Module *RequestingModule,
ModuleMap::KnownHeader *SuggestedModule,
bool IsSystemHeaderDir);

bool findUsableModuleForFrameworkHeader(
const FileEntry *File, StringRef FrameworkDir, Module *RequestingModule,
ModuleMap::KnownHeader *SuggestedModule, bool IsSystemFramework);

const FileEntry *
getFileAndSuggestModule(StringRef FileName, SourceLocation IncludeLoc,
const DirectoryEntry *Dir, bool IsSystemHeaderDir,
Module *RequestingModule,
ModuleMap::KnownHeader *SuggestedModule);

public:
ModuleMap &getModuleMap() { return ModMap; }

const ModuleMap &getModuleMap() const { return ModMap; }

unsigned header_file_size() const { return FileInfo.size(); }

HeaderFileInfo &getFileInfo(const FileEntry *FE);

const HeaderFileInfo *getExistingFileInfo(const FileEntry *FE,
bool WantExternal = true) const;

using search_dir_iterator = std::vector<DirectoryLookup>::const_iterator;

search_dir_iterator search_dir_begin() const { return SearchDirs.begin(); }
search_dir_iterator search_dir_end() const { return SearchDirs.end(); }
unsigned search_dir_size() const { return SearchDirs.size(); }

search_dir_iterator quoted_dir_begin() const {
return SearchDirs.begin();
}

search_dir_iterator quoted_dir_end() const {
return SearchDirs.begin() + AngledDirIdx;
}

search_dir_iterator angled_dir_begin() const {
return SearchDirs.begin() + AngledDirIdx;
}

search_dir_iterator angled_dir_end() const {
return SearchDirs.begin() + SystemDirIdx;
}

search_dir_iterator system_dir_begin() const {
return SearchDirs.begin() + SystemDirIdx;
}

search_dir_iterator system_dir_end() const { return SearchDirs.end(); }

StringRef getUniqueFrameworkName(StringRef Framework);

std::string suggestPathToFileForDiagnostics(const FileEntry *File,
bool *IsSystem = nullptr);

std::string suggestPathToFileForDiagnostics(llvm::StringRef File,
llvm::StringRef WorkingDir,
bool *IsSystem = nullptr);

void PrintStats();

size_t getTotalMemory() const;

private:
enum LoadModuleMapResult {
LMM_AlreadyLoaded,

LMM_NewlyLoaded,

LMM_NoDirectory,

LMM_InvalidModuleMap
};

LoadModuleMapResult loadModuleMapFileImpl(const FileEntry *File,
bool IsSystem,
const DirectoryEntry *Dir,
FileID ID = FileID(),
unsigned *Offset = nullptr);

LoadModuleMapResult loadModuleMapFile(StringRef DirName, bool IsSystem,
bool IsFramework);

LoadModuleMapResult loadModuleMapFile(const DirectoryEntry *Dir,
bool IsSystem, bool IsFramework);
};

} 

#endif 
