
#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSMap.h>
#include <functional>

namespace Aws
{

namespace FileSystem
{
struct DirectoryEntry;
class Directory;

#ifdef _WIN32
static const char PATH_DELIM = '\\';
#else
static const char PATH_DELIM = '/';
#endif


AWS_CORE_API Aws::String GetHomeDirectory();


AWS_CORE_API Aws::String GetExecutableDirectory();


AWS_CORE_API bool CreateDirectoryIfNotExists(const char* path);


AWS_CORE_API bool RemoveDirectoryIfExists(const char* path);


AWS_CORE_API bool RemoveFileIfExists(const char* fileName);


AWS_CORE_API bool RelocateFileOrDirectory(const char* from, const char* to);


AWS_CORE_API bool DeepCopyDirectory(const char* from, const char* to);


AWS_CORE_API bool DeepDeleteDirectory(const char* toDelete);


AWS_CORE_API Aws::String CreateTempFilePath();


AWS_CORE_API std::shared_ptr<Directory> OpenDirectory(const Aws::String& path, const Aws::String& relativePath = "");


AWS_CORE_API Aws::String Join(const Aws::String& leftSegment, const Aws::String& rightSegment);


AWS_CORE_API Aws::String Join(char delimiter, const Aws::String& leftSegment, const Aws::String& rightSegment);


enum class FileType
{
None,
File,
Symlink,
Directory
};

struct DirectoryEntry
{
DirectoryEntry() : fileType(FileType::None), fileSize(0) {}

operator bool() const { return !path.empty() && fileType != FileType::None; }

Aws::String path;
Aws::String relativePath;
FileType fileType;
int64_t fileSize;
};


class AWS_CORE_API Directory
{
public:

Directory(const Aws::String& path, const Aws::String& relativePath);        


virtual operator bool() const { return m_directoryEntry.operator bool(); }


const DirectoryEntry& GetDirectoryEntry() const { return m_directoryEntry; }


const Aws::String& GetPath() const { return m_directoryEntry.path; }


virtual DirectoryEntry Next() = 0;


Directory& Descend(const DirectoryEntry& directoryEntry);


static Aws::Vector<Aws::String> GetAllFilePathsInDirectory(const Aws::String& path);

protected:
DirectoryEntry m_directoryEntry;

private:
Aws::Vector<std::shared_ptr<Directory>> m_openDirectories;
};

class DirectoryTree;


typedef std::function<bool(const DirectoryTree*, const DirectoryEntry&)> DirectoryEntryVisitor;    


class AWS_CORE_API DirectoryTree
{
public:

DirectoryTree(const Aws::String& path);


bool operator==(DirectoryTree& other);


bool operator==(const Aws::String& path);


Aws::Map<Aws::String, DirectoryEntry> Diff(DirectoryTree& other);


operator bool() const;


void TraverseDepthFirst(const DirectoryEntryVisitor& visitor, bool postOrderTraversal = false);


void TraverseBreadthFirst(const DirectoryEntryVisitor& visitor);

private:
bool TraverseDepthFirst(Directory& dir, const DirectoryEntryVisitor& visitor, bool postOrder = false);
void TraverseBreadthFirst(Directory& dir, const DirectoryEntryVisitor& visitor);

std::shared_ptr<Directory> m_dir;
};

} 
} 
