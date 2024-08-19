

#pragma once

#include <string> 
#include <vector> 

namespace rawspeed {

struct ChecksumFileEntry {
std::string FullFileName;
std::string RelFileName;
};

std::vector<ChecksumFileEntry>
ParseChecksumFileContent(const std::string& ChecksumFileContent,
const std::string& RootDir);

std::vector<ChecksumFileEntry>
ReadChecksumFile(const std::string& RootDir,
const std::string& ChecksumFileBasename = "filelist.sha1");

} 
