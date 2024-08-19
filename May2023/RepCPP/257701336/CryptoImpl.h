
#pragma once

#include <aws/core/utils/crypto/Hash.h>
#include <aws/core/utils/crypto/HMAC.h>
#include <aws/core/utils/crypto/Cipher.h>
#include <aws/core/utils/crypto/SecureRandom.h>
#include <aws/core/utils/GetTheLights.h>
#include <openssl/ossl_typ.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/rand.h>
#include <atomic>
#include <mutex>

namespace Aws
{
namespace Utils
{
namespace Crypto
{
namespace OpenSSL
{
extern GetTheLights getTheLights;

void init_static_state();

void cleanup_static_state();

void locking_fn(int mode, int n, const char* file, int line);

unsigned long id_fn();
}


class SecureRandomBytes_OpenSSLImpl : public SecureRandomBytes
{
public:
SecureRandomBytes_OpenSSLImpl()
{ }

~SecureRandomBytes_OpenSSLImpl() = default;


void GetBytes(unsigned char* buffer, size_t bufferSize) override;
};

class MD5OpenSSLImpl : public Hash
{
public:

MD5OpenSSLImpl()
{ }

virtual ~MD5OpenSSLImpl() = default;

virtual HashResult Calculate(const Aws::String& str) override;

virtual HashResult Calculate(Aws::IStream& stream) override;

};

class Sha256OpenSSLImpl : public Hash
{
public:
Sha256OpenSSLImpl()
{ }

virtual ~Sha256OpenSSLImpl() = default;

virtual HashResult Calculate(const Aws::String& str) override;

virtual HashResult Calculate(Aws::IStream& stream) override;
};

class Sha256HMACOpenSSLImpl : public HMAC
{
public:
virtual ~Sha256HMACOpenSSLImpl() = default;

virtual HashResult Calculate(const ByteBuffer& toSign, const ByteBuffer& secret) override;
};


class OpenSSLCipher : public SymmetricCipher
{
public:

OpenSSLCipher(const CryptoBuffer& key, size_t ivSize, bool ctrMode = false);


OpenSSLCipher(CryptoBuffer&& key, CryptoBuffer&& initializationVector,
CryptoBuffer&& tag = CryptoBuffer(0));


OpenSSLCipher(const CryptoBuffer& key, const CryptoBuffer& initializationVector,
const CryptoBuffer& tag = CryptoBuffer(0));

OpenSSLCipher(const OpenSSLCipher& other) = delete;

OpenSSLCipher& operator=(const OpenSSLCipher& other) = delete;


OpenSSLCipher(OpenSSLCipher&& toMove);


OpenSSLCipher& operator=(OpenSSLCipher&& toMove) = default;


virtual ~OpenSSLCipher();


CryptoBuffer EncryptBuffer(const CryptoBuffer& unEncryptedData) override;


CryptoBuffer FinalizeEncryption() override;


CryptoBuffer DecryptBuffer(const CryptoBuffer& encryptedData) override;


CryptoBuffer FinalizeDecryption() override;

void Reset() override;

protected:

virtual void InitEncryptor_Internal() = 0;


virtual void InitDecryptor_Internal() = 0;

virtual size_t GetBlockSizeBytes() const = 0;

virtual size_t GetKeyLengthBits() const = 0;

EVP_CIPHER_CTX* m_ctx;

void CheckInitEncryptor();
void CheckInitDecryptor();

private:
void Init();
void Cleanup();

bool m_encDecInitialized;
bool m_encryptionMode;
bool m_decryptionMode;
};


class AES_CBC_Cipher_OpenSSL : public OpenSSLCipher
{
public:

AES_CBC_Cipher_OpenSSL(const CryptoBuffer& key);


AES_CBC_Cipher_OpenSSL(CryptoBuffer&& key, CryptoBuffer&& initializationVector);


AES_CBC_Cipher_OpenSSL(const CryptoBuffer& key, const CryptoBuffer& initializationVector);

AES_CBC_Cipher_OpenSSL(const AES_CBC_Cipher_OpenSSL& other) = delete;

AES_CBC_Cipher_OpenSSL& operator=(const AES_CBC_Cipher_OpenSSL& other) = delete;

AES_CBC_Cipher_OpenSSL(AES_CBC_Cipher_OpenSSL&& toMove) = default;

protected:
void InitEncryptor_Internal() override;

void InitDecryptor_Internal() override;

size_t GetBlockSizeBytes() const override;

size_t GetKeyLengthBits() const override;

private:
static size_t BlockSizeBytes;
static size_t KeyLengthBits;
};


class AES_CTR_Cipher_OpenSSL : public OpenSSLCipher
{
public:

AES_CTR_Cipher_OpenSSL(const CryptoBuffer& key);


AES_CTR_Cipher_OpenSSL(CryptoBuffer&& key, CryptoBuffer&& initializationVector);


AES_CTR_Cipher_OpenSSL(const CryptoBuffer& key, const CryptoBuffer& initializationVector);

AES_CTR_Cipher_OpenSSL(const AES_CTR_Cipher_OpenSSL& other) = delete;

AES_CTR_Cipher_OpenSSL& operator=(const AES_CTR_Cipher_OpenSSL& other) = delete;

AES_CTR_Cipher_OpenSSL(AES_CTR_Cipher_OpenSSL&& toMove) = default;

protected:
void InitEncryptor_Internal() override;

void InitDecryptor_Internal() override;

size_t GetBlockSizeBytes() const override;

size_t GetKeyLengthBits() const override;

private:
static size_t BlockSizeBytes;
static size_t KeyLengthBits;
};


class AES_GCM_Cipher_OpenSSL : public OpenSSLCipher
{
public:

AES_GCM_Cipher_OpenSSL(const CryptoBuffer& key);


AES_GCM_Cipher_OpenSSL(CryptoBuffer&& key, CryptoBuffer&& initializationVector,
CryptoBuffer&& tag = CryptoBuffer(0));


AES_GCM_Cipher_OpenSSL(const CryptoBuffer& key, const CryptoBuffer& initializationVector,
const CryptoBuffer& tag = CryptoBuffer(0));

AES_GCM_Cipher_OpenSSL(const AES_GCM_Cipher_OpenSSL& other) = delete;

AES_GCM_Cipher_OpenSSL& operator=(const AES_GCM_Cipher_OpenSSL& other) = delete;

AES_GCM_Cipher_OpenSSL(AES_GCM_Cipher_OpenSSL&& toMove) = default;


CryptoBuffer FinalizeEncryption() override;

protected:
void InitEncryptor_Internal() override;

void InitDecryptor_Internal() override;

size_t GetBlockSizeBytes() const override;

size_t GetKeyLengthBits() const override;

size_t GetTagLengthBytes() const;

private:
static size_t BlockSizeBytes;
static size_t IVLengthBytes;
static size_t KeyLengthBits;
static size_t TagLengthBytes;
};


class AES_KeyWrap_Cipher_OpenSSL : public OpenSSLCipher
{
public:


AES_KeyWrap_Cipher_OpenSSL(const CryptoBuffer& key);

AES_KeyWrap_Cipher_OpenSSL(const AES_KeyWrap_Cipher_OpenSSL&) = delete;

AES_KeyWrap_Cipher_OpenSSL& operator=(const AES_KeyWrap_Cipher_OpenSSL&) = delete;

AES_KeyWrap_Cipher_OpenSSL(AES_KeyWrap_Cipher_OpenSSL&&) = default;

CryptoBuffer EncryptBuffer(const CryptoBuffer&) override;
CryptoBuffer FinalizeEncryption() override;

CryptoBuffer DecryptBuffer(const CryptoBuffer&) override;
CryptoBuffer FinalizeDecryption() override;

protected:
void InitEncryptor_Internal() override;

void InitDecryptor_Internal() override;

inline size_t GetBlockSizeBytes() const override { return BlockSizeBytes; }

inline size_t GetKeyLengthBits() const override { return KeyLengthBits; }

private:
static size_t BlockSizeBytes;
static size_t KeyLengthBits;

CryptoBuffer m_workingKeyBuffer;
};

} 
} 
} 
