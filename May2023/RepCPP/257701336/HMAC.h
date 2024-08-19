


#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/Array.h>
#include <aws/core/utils/crypto/HashResult.h>

namespace Aws
{
namespace Utils
{
namespace Crypto
{

class AWS_CORE_API HMAC
{
public:
HMAC() {};
virtual ~HMAC() {};


virtual HashResult Calculate(const Aws::Utils::ByteBuffer& toSign, const Aws::Utils::ByteBuffer& secret) = 0;

};


class AWS_CORE_API HMACFactory
{
public:
virtual ~HMACFactory() {}


virtual std::shared_ptr<HMAC> CreateImplementation() const = 0;


virtual void InitStaticState() {}


virtual void CleanupStaticState() {}
};

} 
} 
} 

