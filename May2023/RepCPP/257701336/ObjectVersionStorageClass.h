

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace S3
{
namespace Model
{
enum class ObjectVersionStorageClass
{
NOT_SET,
STANDARD
};

namespace ObjectVersionStorageClassMapper
{
AWS_S3_API ObjectVersionStorageClass GetObjectVersionStorageClassForName(const Aws::String& name);

AWS_S3_API Aws::String GetNameForObjectVersionStorageClass(ObjectVersionStorageClass value);
} 
} 
} 
} 
