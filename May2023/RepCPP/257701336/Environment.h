

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace Environment
{

AWS_CORE_API Aws::String GetEnv(const char* name);

} 
} 

