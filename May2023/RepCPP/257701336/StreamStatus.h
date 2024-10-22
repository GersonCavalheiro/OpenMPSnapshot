

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace Kinesis
{
namespace Model
{
enum class StreamStatus
{
NOT_SET,
CREATING,
DELETING,
ACTIVE,
UPDATING
};

namespace StreamStatusMapper
{
AWS_KINESIS_API StreamStatus GetStreamStatusForName(const Aws::String& name);

AWS_KINESIS_API Aws::String GetNameForStreamStatus(StreamStatus value);
} 
} 
} 
} 
