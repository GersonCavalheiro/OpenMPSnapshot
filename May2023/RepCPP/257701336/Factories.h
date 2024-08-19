

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/Array.h>
#include <memory>

namespace Aws
{
namespace Utils
{
namespace Crypto
{
class Hash;
class HMAC;
class SymmetricCipher;
class HashFactory;
class HMACFactory;
class SymmetricCipherFactory;
class SecureRandomBytes;
class SecureRandomFactory;


AWS_CORE_API void InitCrypto();

AWS_CORE_API void CleanupCrypto();

AWS_CORE_API void SetInitCleanupOpenSSLFlag(bool initCleanupFlag);


AWS_CORE_API std::shared_ptr<Hash> CreateMD5Implementation();

AWS_CORE_API std::shared_ptr<Hash> CreateSha256Implementation();

AWS_CORE_API std::shared_ptr<HMAC> CreateSha256HMACImplementation();


AWS_CORE_API std::shared_ptr<SymmetricCipher> CreateAES_CBCImplementation(const CryptoBuffer& key);

AWS_CORE_API std::shared_ptr<SymmetricCipher> CreateAES_CBCImplementation(const CryptoBuffer& key, const CryptoBuffer& iv);

AWS_CORE_API std::shared_ptr<SymmetricCipher> CreateAES_CBCImplementation(CryptoBuffer&& key, CryptoBuffer&& iv);


AWS_CORE_API std::shared_ptr<SymmetricCipher> CreateAES_CTRImplementation(const CryptoBuffer& key);

AWS_CORE_API std::shared_ptr<SymmetricCipher> CreateAES_CTRImplementation(const CryptoBuffer& key, const CryptoBuffer& iv);

AWS_CORE_API std::shared_ptr<SymmetricCipher> CreateAES_CTRImplementation(CryptoBuffer&& key, CryptoBuffer&& iv);


AWS_CORE_API std::shared_ptr<SymmetricCipher> CreateAES_GCMImplementation(const CryptoBuffer& key);

AWS_CORE_API std::shared_ptr<SymmetricCipher> CreateAES_GCMImplementation(const CryptoBuffer& key, const CryptoBuffer& iv,
const CryptoBuffer& tag = CryptoBuffer(0));

AWS_CORE_API std::shared_ptr<SymmetricCipher> CreateAES_GCMImplementation(CryptoBuffer&& key, CryptoBuffer&& iv,
CryptoBuffer&& tag = CryptoBuffer(0));

AWS_CORE_API std::shared_ptr<SymmetricCipher> CreateAES_KeyWrapImplementation(const CryptoBuffer& key);   


AWS_CORE_API std::shared_ptr<SecureRandomBytes> CreateSecureRandomBytesImplementation();


AWS_CORE_API void SetMD5Factory(const std::shared_ptr<HashFactory>& factory);

AWS_CORE_API void SetSha256Factory(const std::shared_ptr<HashFactory>& factory);

AWS_CORE_API void SetSha256HMACFactory(const std::shared_ptr<HMACFactory>& factory);

AWS_CORE_API void SetAES_CBCFactory(const std::shared_ptr<SymmetricCipherFactory>& factory);

AWS_CORE_API void SetAES_CTRFactory(const std::shared_ptr<SymmetricCipherFactory>& factory);

AWS_CORE_API void SetAES_GCMFactory(const std::shared_ptr<SymmetricCipherFactory>& factory);

AWS_CORE_API void SetAES_KeyWrapFactory(const std::shared_ptr<SymmetricCipherFactory>& factory);

AWS_CORE_API void SetSecureRandomFactory(const std::shared_ptr<SecureRandomFactory>& factory);

} 
} 
} 

