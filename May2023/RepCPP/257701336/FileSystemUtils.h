
#pragma once
#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSStreamFwd.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/platform/FileSystem.h>

#include <fstream>

namespace Aws
{
namespace Utils
{

class AWS_CORE_API FStreamWithFileName : public Aws::FStream
{
public:
FStreamWithFileName(const Aws::String& fileName, std::ios_base::openmode openFlags) :
Aws::FStream(fileName.c_str(), openFlags), m_fileName(fileName) {}

virtual ~FStreamWithFileName() = default;

const Aws::String& GetFileName() const { return m_fileName; }
protected:
Aws::String m_fileName;
};


class AWS_CORE_API TempFile : public Aws::Utils::FStreamWithFileName
{
public:

TempFile(const char* prefix, const char* suffix, std::ios_base::openmode openFlags);

TempFile(const char* prefix, std::ios_base::openmode openFlags);

TempFile(std::ios_base::openmode openFlags);

~TempFile();
};

class AWS_CORE_API PathUtils
{
public:


static Aws::String GetFileNameFromPathWithoutExt(const Aws::String& path);


static Aws::String GetFileNameFromPathWithExt(const Aws::String& path);
};
}
}
