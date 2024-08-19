
#pragma once

#include <aws/core/Aws.h>
#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/crypto/CryptoBuf.h>
#include <aws/core/utils/crypto/ContentCryptoScheme.h>
#include <aws/core/utils/crypto/KeyWrapAlgorithm.h>

namespace Aws
{
namespace Utils
{
namespace Crypto
{
class AWS_CORE_API ContentCryptoMaterial
{
public:

ContentCryptoMaterial();

ContentCryptoMaterial(ContentCryptoScheme contentCryptoScheme);


ContentCryptoMaterial(const Aws::Utils::CryptoBuffer& cek, ContentCryptoScheme contentCryptoScheme);


inline const Aws::Utils::CryptoBuffer& GetContentEncryptionKey() const
{
return m_contentEncryptionKey;
}


inline const Aws::Utils::CryptoBuffer& GetEncryptedContentEncryptionKey() const
{
return m_encryptedContentEncryptionKey;
}


inline const Aws::Utils::CryptoBuffer& GetIV() const
{
return m_iv;
}


inline size_t GetCryptoTagLength() const
{
return m_cryptoTagLength;
}


inline const Aws::Map<Aws::String, Aws::String>& GetMaterialsDescription() const
{
return m_materialsDescription;
}


inline const Aws::String& GetMaterialsDescription(const Aws::String& key) const
{
return m_materialsDescription.at(key);
}


inline KeyWrapAlgorithm GetKeyWrapAlgorithm() const
{
return m_keyWrapAlgorithm;
}


inline ContentCryptoScheme GetContentCryptoScheme() const
{
return m_contentCryptoScheme;
}


inline void SetContentEncryptionKey(const Aws::Utils::CryptoBuffer& contentEncryptionKey)
{
m_contentEncryptionKey = contentEncryptionKey;
}


inline void SetEncryptedContentEncryptionKey(const Aws::Utils::CryptoBuffer& encryptedContentEncryptionKey)
{
m_encryptedContentEncryptionKey = encryptedContentEncryptionKey;
}


inline void SetIV(const Aws::Utils::CryptoBuffer& iv)
{
m_iv = iv;
}


inline void SetCryptoTagLength(size_t cryptoTagLength)
{
m_cryptoTagLength = cryptoTagLength;
}


inline void AddMaterialsDescription(const Aws::String& key, const Aws::String& value)
{
m_materialsDescription[key] = value;
}


inline void SetMaterialsDescription(const Aws::Map<Aws::String, Aws::String>& materialsDescription)
{
m_materialsDescription = materialsDescription;
}


inline void SetKeyWrapAlgorithm(KeyWrapAlgorithm keyWrapAlgorithm)
{
m_keyWrapAlgorithm = keyWrapAlgorithm;
}


inline void SetContentCryptoScheme(ContentCryptoScheme contentCryptoScheme)
{
m_contentCryptoScheme = contentCryptoScheme;
}

private:
Aws::Utils::CryptoBuffer m_contentEncryptionKey;
Aws::Utils::CryptoBuffer m_encryptedContentEncryptionKey;
Aws::Utils::CryptoBuffer m_iv;
size_t m_cryptoTagLength;
Aws::Map<Aws::String, Aws::String> m_materialsDescription;
KeyWrapAlgorithm m_keyWrapAlgorithm;
ContentCryptoScheme m_contentCryptoScheme;
};
}
}
}