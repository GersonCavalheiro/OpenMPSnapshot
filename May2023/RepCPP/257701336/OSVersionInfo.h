

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace OSVersionInfo
{

AWS_CORE_API Aws::String ComputeOSVersionString();


AWS_CORE_API Aws::String GetSysCommandOutput(const char* command);

} 
} 

