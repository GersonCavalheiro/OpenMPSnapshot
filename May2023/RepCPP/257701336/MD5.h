


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

#include <aws/core/utils/crypto/Hash.h>
#include <aws/core/utils/Outcome.h>

namespace Aws
{
namespace Utils
{
namespace Crypto
{
class WindowsHashImpl;


class AWS_CORE_API MD5 : public Hash
{
public:

MD5();
virtual ~MD5();


virtual HashResult Calculate(const Aws::String& str) override;


virtual HashResult Calculate(Aws::IStream& stream) override;

private:

std::shared_ptr<Hash> m_hashImpl;
};

} 
} 
} 

