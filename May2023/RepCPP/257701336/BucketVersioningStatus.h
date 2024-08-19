

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace S3
{
namespace Model
{
enum class BucketVersioningStatus
{
NOT_SET,
Enabled,
Suspended
};

namespace BucketVersioningStatusMapper
{
AWS_S3_API BucketVersioningStatus GetBucketVersioningStatusForName(const Aws::String& name);

AWS_S3_API Aws::String GetNameForBucketVersioningStatus(BucketVersioningStatus value);
} 
} 
} 
} 
