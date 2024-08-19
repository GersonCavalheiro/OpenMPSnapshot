

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace S3
{
namespace Model
{
enum class ObjectStorageClass
{
NOT_SET,
STANDARD,
REDUCED_REDUNDANCY,
GLACIER
};

namespace ObjectStorageClassMapper
{
AWS_S3_API ObjectStorageClass GetObjectStorageClassForName(const Aws::String& name);

AWS_S3_API Aws::String GetNameForObjectStorageClass(ObjectStorageClass value);
} 
} 
} 
} 
