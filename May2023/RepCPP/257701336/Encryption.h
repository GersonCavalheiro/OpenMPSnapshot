

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/ServerSideEncryption.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <utility>

namespace Aws
{
namespace Utils
{
namespace Xml
{
class XmlNode;
} 
} 
namespace S3
{
namespace Model
{


class AWS_S3_API Encryption
{
public:
Encryption();
Encryption(const Aws::Utils::Xml::XmlNode& xmlNode);
Encryption& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const ServerSideEncryption& GetEncryptionType() const{ return m_encryptionType; }


inline void SetEncryptionType(const ServerSideEncryption& value) { m_encryptionTypeHasBeenSet = true; m_encryptionType = value; }


inline void SetEncryptionType(ServerSideEncryption&& value) { m_encryptionTypeHasBeenSet = true; m_encryptionType = std::move(value); }


inline Encryption& WithEncryptionType(const ServerSideEncryption& value) { SetEncryptionType(value); return *this;}


inline Encryption& WithEncryptionType(ServerSideEncryption&& value) { SetEncryptionType(std::move(value)); return *this;}



inline const Aws::String& GetKMSKeyId() const{ return m_kMSKeyId; }


inline void SetKMSKeyId(const Aws::String& value) { m_kMSKeyIdHasBeenSet = true; m_kMSKeyId = value; }


inline void SetKMSKeyId(Aws::String&& value) { m_kMSKeyIdHasBeenSet = true; m_kMSKeyId = std::move(value); }


inline void SetKMSKeyId(const char* value) { m_kMSKeyIdHasBeenSet = true; m_kMSKeyId.assign(value); }


inline Encryption& WithKMSKeyId(const Aws::String& value) { SetKMSKeyId(value); return *this;}


inline Encryption& WithKMSKeyId(Aws::String&& value) { SetKMSKeyId(std::move(value)); return *this;}


inline Encryption& WithKMSKeyId(const char* value) { SetKMSKeyId(value); return *this;}



inline const Aws::String& GetKMSContext() const{ return m_kMSContext; }


inline void SetKMSContext(const Aws::String& value) { m_kMSContextHasBeenSet = true; m_kMSContext = value; }


inline void SetKMSContext(Aws::String&& value) { m_kMSContextHasBeenSet = true; m_kMSContext = std::move(value); }


inline void SetKMSContext(const char* value) { m_kMSContextHasBeenSet = true; m_kMSContext.assign(value); }


inline Encryption& WithKMSContext(const Aws::String& value) { SetKMSContext(value); return *this;}


inline Encryption& WithKMSContext(Aws::String&& value) { SetKMSContext(std::move(value)); return *this;}


inline Encryption& WithKMSContext(const char* value) { SetKMSContext(value); return *this;}

private:

ServerSideEncryption m_encryptionType;
bool m_encryptionTypeHasBeenSet;

Aws::String m_kMSKeyId;
bool m_kMSKeyIdHasBeenSet;

Aws::String m_kMSContext;
bool m_kMSContextHasBeenSet;
};

} 
} 
} 
