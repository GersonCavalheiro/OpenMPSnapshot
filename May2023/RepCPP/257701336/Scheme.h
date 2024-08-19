

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace Http
{

enum class Scheme
{
HTTP,
HTTPS
};

namespace SchemeMapper
{

AWS_CORE_API const char* ToString(Scheme scheme);

AWS_CORE_API Scheme FromString(const char* name);
} 
} 
} 

