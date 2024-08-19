

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace S3
{
namespace Model
{
enum class ExpirationStatus
{
NOT_SET,
Enabled,
Disabled
};

namespace ExpirationStatusMapper
{
AWS_S3_API ExpirationStatus GetExpirationStatusForName(const Aws::String& name);

AWS_S3_API Aws::String GetNameForExpirationStatus(ExpirationStatus value);
} 
} 
} 
} 
