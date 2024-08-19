
#pragma once

#include <aws/core/utils/crypto/CryptoBuf.h>
#include <aws/core/Core_EXPORTS.h>

namespace Aws
{
namespace Utils
{
namespace Crypto
{

class AWS_CORE_API SymmetricCryptoStream : public Aws::IOStream
{
public:

SymmetricCryptoStream(Aws::IStream& src, CipherMode mode, SymmetricCipher& cipher, size_t bufLen = DEFAULT_BUF_SIZE);

SymmetricCryptoStream(Aws::OStream& sink, CipherMode mode, SymmetricCipher& cipher, size_t bufLen = DEFAULT_BUF_SIZE, int16_t blockOffset = 0 );

SymmetricCryptoStream(Aws::Utils::Crypto::SymmetricCryptoBufSrc& bufSrc);

SymmetricCryptoStream(Aws::Utils::Crypto::SymmetricCryptoBufSink& bufSink);

SymmetricCryptoStream(const SymmetricCryptoStream&) = delete;
SymmetricCryptoStream(SymmetricCryptoStream&&) = delete;

virtual ~SymmetricCryptoStream();

SymmetricCryptoStream& operator=(const SymmetricCryptoStream&) = delete;
SymmetricCryptoStream& operator=(SymmetricCryptoStream&&) = delete;



void Finalize();

private:
CryptoBuf* m_cryptoBuf;
bool m_hasOwnership;
};
}
}
}