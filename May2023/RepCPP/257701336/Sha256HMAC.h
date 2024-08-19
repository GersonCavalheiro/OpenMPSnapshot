


#pragma once

#ifdef __APPLE__

#ifdef __clang__
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif 

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif 

#endif 

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/crypto/HMAC.h>
#include <aws/core/utils/memory/AWSMemory.h>

namespace Aws
{
namespace Utils
{
namespace Crypto
{

class AWS_CORE_API Sha256HMAC : public HMAC
{
public:

Sha256HMAC();
virtual ~Sha256HMAC();


virtual HashResult Calculate(const Aws::Utils::ByteBuffer& toSign, const Aws::Utils::ByteBuffer& secret) override;

private:

std::shared_ptr< HMAC > m_hmacImpl;
};

} 
} 
} 

