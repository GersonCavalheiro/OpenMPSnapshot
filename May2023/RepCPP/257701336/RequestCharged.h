

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace S3
{
namespace Model
{
enum class RequestCharged
{
NOT_SET,
requester
};

namespace RequestChargedMapper
{
AWS_S3_API RequestCharged GetRequestChargedForName(const Aws::String& name);

AWS_S3_API Aws::String GetNameForRequestCharged(RequestCharged value);
} 
} 
} 
} 
