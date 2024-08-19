

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/Array.h>

namespace Aws
{
namespace Utils
{

template< typename R, typename E > class Outcome;

namespace Crypto
{
using HashResult = Outcome< ByteBuffer, bool >;

} 
} 
} 

