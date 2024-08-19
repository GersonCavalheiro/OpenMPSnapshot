

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace S3
{
namespace Model
{
enum class StorageClass
{
NOT_SET,
STANDARD,
REDUCED_REDUNDANCY,
STANDARD_IA
};

namespace StorageClassMapper
{
AWS_S3_API StorageClass GetStorageClassForName(const Aws::String& name);

AWS_S3_API Aws::String GetNameForStorageClass(StorageClass value);
} 
} 
} 
} 
