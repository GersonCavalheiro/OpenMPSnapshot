

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/crypto/HashResult.h>
#include <aws/core/utils/memory/stl/AWSStreamFwd.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace Utils
{
namespace Crypto
{

class AWS_CORE_API Hash
{
public:

Hash() {}
virtual ~Hash() {}


virtual HashResult Calculate(const Aws::String& str) = 0;


virtual HashResult Calculate(Aws::IStream& stream) = 0;

static const uint32_t INTERNAL_HASH_STREAM_BUFFER_SIZE = 8192;
};


class AWS_CORE_API HashFactory
{
public:
virtual ~HashFactory() {}


virtual std::shared_ptr<Hash> CreateImplementation() const = 0;


virtual void InitStaticState() {}


virtual void CleanupStaticState() {}
};

} 
} 
} 

