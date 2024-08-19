

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <stddef.h>

namespace Aws
{
namespace Security
{


AWS_CORE_API void SecureMemClear(unsigned char *data, size_t length);

} 
} 

