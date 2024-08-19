

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace Kinesis
{
namespace Model
{
enum class EncryptionType
{
NOT_SET,
NONE,
KMS
};

namespace EncryptionTypeMapper
{
AWS_KINESIS_API EncryptionType GetEncryptionTypeForName(const Aws::String& name);

AWS_KINESIS_API Aws::String GetNameForEncryptionType(EncryptionType value);
} 
} 
} 
} 
