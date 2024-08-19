

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace S3
{
namespace Model
{
enum class RequestPayer
{
NOT_SET,
requester
};

namespace RequestPayerMapper
{
AWS_S3_API RequestPayer GetRequestPayerForName(const Aws::String& name);

AWS_S3_API Aws::String GetNameForRequestPayer(RequestPayer value);
} 
} 
} 
} 
