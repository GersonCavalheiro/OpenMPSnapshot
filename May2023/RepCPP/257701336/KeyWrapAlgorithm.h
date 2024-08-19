
#pragma once
#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace Utils
{
namespace Crypto
{
enum class KeyWrapAlgorithm
{
KMS,
AES_KEY_WRAP,
NONE
};

namespace KeyWrapAlgorithmMapper
{
AWS_CORE_API KeyWrapAlgorithm GetKeyWrapAlgorithmForName(const Aws::String& name);

AWS_CORE_API Aws::String GetNameForKeyWrapAlgorithm(KeyWrapAlgorithm enumValue);
}
} 

}
}
