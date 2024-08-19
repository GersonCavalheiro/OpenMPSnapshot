

#pragma once

#include <aws/core/Core_EXPORTS.h>

namespace Aws
{
namespace Utils
{
class EnumParseOverflowContainer;
}

AWS_CORE_API Utils::EnumParseOverflowContainer* GetEnumOverflowContainer();


AWS_CORE_API bool CheckAndSwapEnumOverflowContainer(Utils::EnumParseOverflowContainer* expectedValue, Utils::EnumParseOverflowContainer* newValue);
}