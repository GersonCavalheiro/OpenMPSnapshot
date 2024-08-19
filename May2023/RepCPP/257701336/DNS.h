

#pragma once

#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace Utils
{
AWS_CORE_API bool IsValidDnsLabel(const Aws::String& label);
}
}
