

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/Array.h>
#include <aws/core/utils/memory/stl/AWSStreamFwd.h>

namespace Aws
{
namespace Utils
{
namespace Crypto
{
static const size_t SYMMETRIC_KEY_LENGTH = 32;
static const size_t MIN_IV_LENGTH = 12;

AWS_CORE_API CryptoBuffer IncrementCTRCounter(const CryptoBuffer& counter, uint32_t numberOfBlocks);


class AWS_CORE_API SymmetricCipher
{
public:

SymmetricCipher(const CryptoBuffer& key, size_t ivSize, bool ivGenerationInCtrMode = false) :
m_key(key), m_initializationVector(ivSize > 0 ? GenerateIV(ivSize, ivGenerationInCtrMode) : 0), m_failure(false) { Validate(); }


SymmetricCipher(const CryptoBuffer& key, const CryptoBuffer& initializationVector, const CryptoBuffer& tag = CryptoBuffer(0)) :
m_key(key), m_initializationVector(initializationVector), m_tag(tag), m_failure(false) { Validate(); }


SymmetricCipher(CryptoBuffer&& key, CryptoBuffer&& initializationVector, CryptoBuffer&& tag = CryptoBuffer(0)) :
m_key(std::move(key)), m_initializationVector(std::move(initializationVector)), m_tag(std::move(tag)), m_failure(false) { Validate(); }

SymmetricCipher(const SymmetricCipher& other) = delete;
SymmetricCipher& operator=(const SymmetricCipher& other) = delete;


SymmetricCipher(SymmetricCipher&& toMove) :
m_key(std::move(toMove.m_key)),
m_initializationVector(std::move(toMove.m_initializationVector)),
m_tag(std::move(toMove.m_tag)),
m_failure(toMove.m_failure)
{
Validate();
}


SymmetricCipher& operator=(SymmetricCipher&& toMove)
{
m_key = std::move(toMove.m_key);
m_initializationVector = std::move(toMove.m_initializationVector);
m_tag = std::move(toMove.m_tag);
m_failure = toMove.m_failure;

Validate();

return *this;
}

virtual ~SymmetricCipher() = default;


virtual operator bool() const { return Good(); }


virtual CryptoBuffer EncryptBuffer( const CryptoBuffer& unEncryptedData) = 0;


virtual CryptoBuffer FinalizeEncryption () = 0;


virtual CryptoBuffer DecryptBuffer(const CryptoBuffer& encryptedData) = 0;


virtual CryptoBuffer FinalizeDecryption () = 0;

virtual void Reset() = 0;


inline const CryptoBuffer& GetIV() const { return m_initializationVector; }


inline const CryptoBuffer& GetTag() const { return m_tag; }

inline bool Fail() const { return m_failure; }
inline bool Good() const { return !Fail(); }


static CryptoBuffer GenerateIV(size_t ivLengthBytes, bool ctrMode = false);


static CryptoBuffer GenerateKey(size_t keyLengthBytes = SYMMETRIC_KEY_LENGTH);                

protected:
SymmetricCipher() : m_failure(false) {}

CryptoBuffer m_key;
CryptoBuffer m_initializationVector;
CryptoBuffer m_tag;
bool m_failure;

private:
void Validate();
};


class SymmetricCipherFactory
{
public:
virtual ~SymmetricCipherFactory() {}


virtual std::shared_ptr<SymmetricCipher> CreateImplementation(const CryptoBuffer& key) const = 0;

virtual std::shared_ptr<SymmetricCipher> CreateImplementation(const CryptoBuffer& key, const CryptoBuffer& iv, const CryptoBuffer& tag = CryptoBuffer(0)) const = 0;

virtual std::shared_ptr<SymmetricCipher> CreateImplementation(CryptoBuffer&& key, CryptoBuffer&& iv, CryptoBuffer&& tag = CryptoBuffer(0)) const = 0;


virtual void InitStaticState() {}


virtual void CleanupStaticState() {}
};
}
}
}