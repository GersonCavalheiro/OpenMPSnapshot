

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/Region.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{

namespace S3
{
namespace S3Endpoint
{
AWS_S3_API Aws::String ForRegion(const Aws::String& regionName, bool useDualStack = false);
} 
} 
} 
