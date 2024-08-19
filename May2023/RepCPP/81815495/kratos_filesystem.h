
#pragma once

#include <string>
#include <vector>
#if __has_include(<filesystem>) 
#include <filesystem> 
#elif __has_include(<experimental/filesystem>) 
#include <experimental/filesystem>
namespace std {
namespace filesystem = experimental::filesystem;
}
#else 
#error Could not find system header "<filesystem>" or "<experimental/filesystem>"
#endif 


#include "includes/define.h"

namespace Kratos {

#if defined(__GNUG__) && __GNUC__ < 10 && !defined(__clang__) && !defined(__INTEL_COMPILER)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif

namespace KRATOS_DEPRECATED_MESSAGE("Please use std::filesystem directly") filesystem {

#if defined(__GNUG__) && __GNUC__ < 10 && !defined(__clang__) && !defined(__INTEL_COMPILER)
#pragma GCC diagnostic pop
#endif

bool KRATOS_API(KRATOS_CORE) exists(const std::string& rPath);

bool KRATOS_API(KRATOS_CORE) is_regular_file(const std::string& rPath);

bool KRATOS_API(KRATOS_CORE) is_directory(const std::string& rPath);

bool KRATOS_API(KRATOS_CORE) create_directory(const std::string& rPath);

bool KRATOS_API(KRATOS_CORE) create_directories(const std::string& rPath);

bool KRATOS_API(KRATOS_CORE) remove(const std::string& rPath);

std::uintmax_t KRATOS_API(KRATOS_CORE) remove_all(const std::string& rPath);

void KRATOS_API(KRATOS_CORE) rename(const std::string& rPathFrom, const std::string& rPathTo);

std::string KRATOS_API(KRATOS_CORE) parent_path(const std::string& rPath);

std::string KRATOS_API(KRATOS_CORE) filename(const std::string& rPath);

} 


class KRATOS_API(KRATOS_CORE) FilesystemExtensions
{

public:
FilesystemExtensions() = delete;

FilesystemExtensions(FilesystemExtensions const& rOther) = delete;

FilesystemExtensions& operator=(FilesystemExtensions const& rOther) = delete;



KRATOS_DEPRECATED_MESSAGE("Please use std::filesystem directly")
static std::string CurrentWorkingDirectory();


KRATOS_DEPRECATED_MESSAGE("Please use the /-operator directly")
static std::string JoinPaths(const std::vector<std::string>& rPaths);


[[nodiscard]] static std::vector<std::filesystem::path> ListDirectory(const std::filesystem::path& rPath);


static void MPISafeCreateDirectories(const std::filesystem::path& rPath);


[[nodiscard]] static std::filesystem::path ResolveSymlinks(const std::filesystem::path& rPath);

}; 

} 
