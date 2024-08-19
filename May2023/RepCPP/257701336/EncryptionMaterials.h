
#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/crypto/CryptoBuf.h>
#include <aws/core/utils/crypto/ContentCryptoMaterial.h>

namespace Aws
{
namespace Utils
{
namespace Crypto
{
class AWS_CORE_API EncryptionMaterials
{
public:
virtual ~EncryptionMaterials();


virtual void EncryptCEK(ContentCryptoMaterial& contentCryptoMaterial) = 0;


virtual void DecryptCEK(ContentCryptoMaterial& contentCryptoMaterial) = 0;
};
}
}
}
