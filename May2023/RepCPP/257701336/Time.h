

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <ctime>

namespace Aws
{
namespace Time
{


AWS_CORE_API time_t TimeGM(tm* const t);


AWS_CORE_API void LocalTime(tm* t, std::time_t time);


AWS_CORE_API void GMTime(tm* t, std::time_t time);

} 
} 
