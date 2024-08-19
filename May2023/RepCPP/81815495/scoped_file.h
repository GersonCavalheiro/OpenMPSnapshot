

#pragma once

#include <filesystem>
#include <fstream>


namespace Kratos::Testing {



class ScopedEntry
{
public:
ScopedEntry(const std::filesystem::path& rPath);

ScopedEntry(ScopedEntry&& rOther) = default;

ScopedEntry(const ScopedEntry& rOther) = delete;

ScopedEntry& operator=(ScopedEntry&& rOther) = delete;

ScopedEntry& operator=(const ScopedEntry& rOther) = delete;

virtual ~ScopedEntry();

operator const std::filesystem::path& () const;

private:
const std::filesystem::path mPath;
}; 


struct ScopedDirectory final : public ScopedEntry
{
ScopedDirectory(const std::filesystem::path& rPath);
}; 


class ScopedFile final : public ScopedEntry
{
public:
ScopedFile(const std::filesystem::path& rPath);

~ScopedFile() override;

template <class T>
friend ScopedFile& operator<<(ScopedFile& rFile, const T& rContent)
{
rFile.mStream << rContent;
rFile.mStream.flush();
return rFile;
}

private:
std::ofstream mStream;
}; 


struct ScopedSymlink final : public ScopedEntry
{
ScopedSymlink(const std::filesystem::path& rSymlinkPath, const std::filesystem::path& rTargetPath);
}; 


} 
