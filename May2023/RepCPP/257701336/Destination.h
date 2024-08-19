

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/StorageClass.h>
#include <aws/s3/model/AccessControlTranslation.h>
#include <aws/s3/model/EncryptionConfiguration.h>
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


class AWS_S3_API Destination
{
public:
Destination();
Destination(const Aws::Utils::Xml::XmlNode& xmlNode);
Destination& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline Destination& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline Destination& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline Destination& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetAccount() const{ return m_account; }


inline void SetAccount(const Aws::String& value) { m_accountHasBeenSet = true; m_account = value; }


inline void SetAccount(Aws::String&& value) { m_accountHasBeenSet = true; m_account = std::move(value); }


inline void SetAccount(const char* value) { m_accountHasBeenSet = true; m_account.assign(value); }


inline Destination& WithAccount(const Aws::String& value) { SetAccount(value); return *this;}


inline Destination& WithAccount(Aws::String&& value) { SetAccount(std::move(value)); return *this;}


inline Destination& WithAccount(const char* value) { SetAccount(value); return *this;}



inline const StorageClass& GetStorageClass() const{ return m_storageClass; }


inline void SetStorageClass(const StorageClass& value) { m_storageClassHasBeenSet = true; m_storageClass = value; }


inline void SetStorageClass(StorageClass&& value) { m_storageClassHasBeenSet = true; m_storageClass = std::move(value); }


inline Destination& WithStorageClass(const StorageClass& value) { SetStorageClass(value); return *this;}


inline Destination& WithStorageClass(StorageClass&& value) { SetStorageClass(std::move(value)); return *this;}



inline const AccessControlTranslation& GetAccessControlTranslation() const{ return m_accessControlTranslation; }


inline void SetAccessControlTranslation(const AccessControlTranslation& value) { m_accessControlTranslationHasBeenSet = true; m_accessControlTranslation = value; }


inline void SetAccessControlTranslation(AccessControlTranslation&& value) { m_accessControlTranslationHasBeenSet = true; m_accessControlTranslation = std::move(value); }


inline Destination& WithAccessControlTranslation(const AccessControlTranslation& value) { SetAccessControlTranslation(value); return *this;}


inline Destination& WithAccessControlTranslation(AccessControlTranslation&& value) { SetAccessControlTranslation(std::move(value)); return *this;}



inline const EncryptionConfiguration& GetEncryptionConfiguration() const{ return m_encryptionConfiguration; }


inline void SetEncryptionConfiguration(const EncryptionConfiguration& value) { m_encryptionConfigurationHasBeenSet = true; m_encryptionConfiguration = value; }


inline void SetEncryptionConfiguration(EncryptionConfiguration&& value) { m_encryptionConfigurationHasBeenSet = true; m_encryptionConfiguration = std::move(value); }


inline Destination& WithEncryptionConfiguration(const EncryptionConfiguration& value) { SetEncryptionConfiguration(value); return *this;}


inline Destination& WithEncryptionConfiguration(EncryptionConfiguration&& value) { SetEncryptionConfiguration(std::move(value)); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_account;
bool m_accountHasBeenSet;

StorageClass m_storageClass;
bool m_storageClassHasBeenSet;

AccessControlTranslation m_accessControlTranslation;
bool m_accessControlTranslationHasBeenSet;

EncryptionConfiguration m_encryptionConfiguration;
bool m_encryptionConfigurationHasBeenSet;
};

} 
} 
} 
