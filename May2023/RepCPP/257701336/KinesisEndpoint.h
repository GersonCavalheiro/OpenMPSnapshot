

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/Region.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{

namespace Kinesis
{
namespace KinesisEndpoint
{
AWS_KINESIS_API Aws::String ForRegion(const Aws::String& regionName, bool useDualStack = false);
} 
} 
} 
