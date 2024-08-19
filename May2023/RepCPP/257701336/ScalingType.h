

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace Kinesis
{
namespace Model
{
enum class ScalingType
{
NOT_SET,
UNIFORM_SCALING
};

namespace ScalingTypeMapper
{
AWS_KINESIS_API ScalingType GetScalingTypeForName(const Aws::String& name);

AWS_KINESIS_API Aws::String GetNameForScalingType(ScalingType value);
} 
} 
} 
} 
