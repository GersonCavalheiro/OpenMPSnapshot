

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace S3
{
namespace Model
{
enum class FileHeaderInfo
{
NOT_SET,
USE,
IGNORE,
NONE
};

namespace FileHeaderInfoMapper
{
AWS_S3_API FileHeaderInfo GetFileHeaderInfoForName(const Aws::String& name);

AWS_S3_API Aws::String GetNameForFileHeaderInfo(FileHeaderInfo value);
} 
} 
} 
} 
