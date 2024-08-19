
#pragma once
#include <type_traits>

namespace Aws
{
namespace Utils
{
namespace Crypto
{

class SecureRandomBytes
{
public:
SecureRandomBytes() : m_failure(false)
{
}

virtual ~SecureRandomBytes() = default;


virtual void GetBytes(unsigned char* buffer, size_t bufferSize) = 0;


operator bool() const { return !m_failure; }

protected:
bool m_failure;
};


template <typename DataType = uint64_t>
class SecureRandom
{
public:

SecureRandom(const std::shared_ptr<SecureRandomBytes>& entropySource) : m_entropy(entropySource)
{ static_assert(std::is_unsigned<DataType>::value, "Type DataType must be integral"); }

virtual ~SecureRandom() = default;

virtual void Reset() {}


virtual DataType operator()()
{
DataType value(0);
unsigned char buffer[sizeof(DataType)];
m_entropy->GetBytes(buffer, sizeof(DataType));

assert(*m_entropy);
if(*m_entropy)
{
for (size_t i = 0; i < sizeof(DataType); ++i)
{
value <<= 8;
value |= buffer[i];

}
}

return value;
}

operator bool() const { return *m_entropy; }

private:
std::shared_ptr<SecureRandomBytes> m_entropy;
};           

class SecureRandomFactory
{
public:

virtual std::shared_ptr<SecureRandomBytes> CreateImplementation() const = 0;


virtual void InitStaticState() {}


virtual void CleanupStaticState() {}
};
}
}
}