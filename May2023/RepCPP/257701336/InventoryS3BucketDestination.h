

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/InventoryFormat.h>
#include <aws/s3/model/InventoryEncryption.h>
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

class AWS_S3_API InventoryS3BucketDestination
{
public:
InventoryS3BucketDestination();
InventoryS3BucketDestination(const Aws::Utils::Xml::XmlNode& xmlNode);
InventoryS3BucketDestination& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetAccountId() const{ return m_accountId; }


inline void SetAccountId(const Aws::String& value) { m_accountIdHasBeenSet = true; m_accountId = value; }


inline void SetAccountId(Aws::String&& value) { m_accountIdHasBeenSet = true; m_accountId = std::move(value); }


inline void SetAccountId(const char* value) { m_accountIdHasBeenSet = true; m_accountId.assign(value); }


inline InventoryS3BucketDestination& WithAccountId(const Aws::String& value) { SetAccountId(value); return *this;}


inline InventoryS3BucketDestination& WithAccountId(Aws::String&& value) { SetAccountId(std::move(value)); return *this;}


inline InventoryS3BucketDestination& WithAccountId(const char* value) { SetAccountId(value); return *this;}



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline InventoryS3BucketDestination& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline InventoryS3BucketDestination& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline InventoryS3BucketDestination& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const InventoryFormat& GetFormat() const{ return m_format; }


inline void SetFormat(const InventoryFormat& value) { m_formatHasBeenSet = true; m_format = value; }


inline void SetFormat(InventoryFormat&& value) { m_formatHasBeenSet = true; m_format = std::move(value); }


inline InventoryS3BucketDestination& WithFormat(const InventoryFormat& value) { SetFormat(value); return *this;}


inline InventoryS3BucketDestination& WithFormat(InventoryFormat&& value) { SetFormat(std::move(value)); return *this;}



inline const Aws::String& GetPrefix() const{ return m_prefix; }


inline void SetPrefix(const Aws::String& value) { m_prefixHasBeenSet = true; m_prefix = value; }


inline void SetPrefix(Aws::String&& value) { m_prefixHasBeenSet = true; m_prefix = std::move(value); }


inline void SetPrefix(const char* value) { m_prefixHasBeenSet = true; m_prefix.assign(value); }


inline InventoryS3BucketDestination& WithPrefix(const Aws::String& value) { SetPrefix(value); return *this;}


inline InventoryS3BucketDestination& WithPrefix(Aws::String&& value) { SetPrefix(std::move(value)); return *this;}


inline InventoryS3BucketDestination& WithPrefix(const char* value) { SetPrefix(value); return *this;}



inline const InventoryEncryption& GetEncryption() const{ return m_encryption; }


inline void SetEncryption(const InventoryEncryption& value) { m_encryptionHasBeenSet = true; m_encryption = value; }


inline void SetEncryption(InventoryEncryption&& value) { m_encryptionHasBeenSet = true; m_encryption = std::move(value); }


inline InventoryS3BucketDestination& WithEncryption(const InventoryEncryption& value) { SetEncryption(value); return *this;}


inline InventoryS3BucketDestination& WithEncryption(InventoryEncryption&& value) { SetEncryption(std::move(value)); return *this;}

private:

Aws::String m_accountId;
bool m_accountIdHasBeenSet;

Aws::String m_bucket;
bool m_bucketHasBeenSet;

InventoryFormat m_format;
bool m_formatHasBeenSet;

Aws::String m_prefix;
bool m_prefixHasBeenSet;

InventoryEncryption m_encryption;
bool m_encryptionHasBeenSet;
};

} 
} 
} 
