
#pragma once

#include <aws/core/utils/crypto/Cipher.h>
#include <aws/core/Core_EXPORTS.h>
#include <ios>

namespace Aws
{
namespace Utils
{
namespace Crypto
{
typedef std::mbstate_t FPOS_TYPE;
static const size_t DEFAULT_BUF_SIZE = 1024;
static const size_t PUT_BACK_SIZE = 1;


enum class CipherMode
{
Encrypt,
Decrypt
};


class AWS_CORE_API CryptoBuf : public std::streambuf
{
public:
CryptoBuf() = default;
virtual ~CryptoBuf() = default;
CryptoBuf(const CryptoBuf&) = delete;
CryptoBuf(CryptoBuf&& rhs) = delete;

virtual void Finalize() {}
};


class AWS_CORE_API SymmetricCryptoBufSrc : public CryptoBuf
{
public:

SymmetricCryptoBufSrc(Aws::IStream& stream, SymmetricCipher& cipher, CipherMode cipherMode, size_t bufferSize = DEFAULT_BUF_SIZE);

SymmetricCryptoBufSrc(const SymmetricCryptoBufSrc&) = delete;
SymmetricCryptoBufSrc(SymmetricCryptoBufSrc&&) = delete;

SymmetricCryptoBufSrc& operator=(const SymmetricCryptoBufSrc&) = delete;
SymmetricCryptoBufSrc& operator=(SymmetricCryptoBufSrc&&) = delete;

virtual ~SymmetricCryptoBufSrc() { FinalizeCipher(); }


void Finalize() override { FinalizeCipher(); }

protected:
pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out ) override;
pos_type seekpos(pos_type pos, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out ) override;

private:
int_type underflow() override;
off_type ComputeAbsSeekPosition(off_type, std::ios_base::seekdir,  std::fpos<FPOS_TYPE>);
void FinalizeCipher();

CryptoBuffer m_isBuf;
SymmetricCipher& m_cipher;
Aws::IStream& m_stream;
CipherMode m_cipherMode;
bool m_isFinalized;
size_t m_bufferSize;
size_t m_putBack;
};


class AWS_CORE_API SymmetricCryptoBufSink : public CryptoBuf
{
public:

SymmetricCryptoBufSink(Aws::OStream& stream, SymmetricCipher& cipher, CipherMode cipherMode, size_t bufferSize = DEFAULT_BUF_SIZE, int16_t blockOffset = 0);
SymmetricCryptoBufSink(const SymmetricCryptoBufSink&) = delete;
SymmetricCryptoBufSink(SymmetricCryptoBufSink&&) = delete;

SymmetricCryptoBufSink& operator=(const SymmetricCryptoBufSink&) = delete;

SymmetricCryptoBufSink& operator=(SymmetricCryptoBufSink&&) = delete;

virtual ~SymmetricCryptoBufSink();


void FinalizeCiphersAndFlushSink();

void Finalize() override { FinalizeCiphersAndFlushSink(); }

private:
int_type overflow(int_type ch) override;
int sync() override;
bool writeOutput(bool finalize);

CryptoBuffer m_osBuf;
SymmetricCipher& m_cipher;
Aws::OStream& m_stream;
CipherMode m_cipherMode;
bool m_isFinalized;
int16_t m_blockOffset;
};
}
}
}
