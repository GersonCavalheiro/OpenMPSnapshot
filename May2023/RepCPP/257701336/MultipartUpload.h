

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/DateTime.h>
#include <aws/s3/model/StorageClass.h>
#include <aws/s3/model/Owner.h>
#include <aws/s3/model/Initiator.h>
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

class AWS_S3_API MultipartUpload
{
public:
MultipartUpload();
MultipartUpload(const Aws::Utils::Xml::XmlNode& xmlNode);
MultipartUpload& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetUploadId() const{ return m_uploadId; }


inline void SetUploadId(const Aws::String& value) { m_uploadIdHasBeenSet = true; m_uploadId = value; }


inline void SetUploadId(Aws::String&& value) { m_uploadIdHasBeenSet = true; m_uploadId = std::move(value); }


inline void SetUploadId(const char* value) { m_uploadIdHasBeenSet = true; m_uploadId.assign(value); }


inline MultipartUpload& WithUploadId(const Aws::String& value) { SetUploadId(value); return *this;}


inline MultipartUpload& WithUploadId(Aws::String&& value) { SetUploadId(std::move(value)); return *this;}


inline MultipartUpload& WithUploadId(const char* value) { SetUploadId(value); return *this;}



inline const Aws::String& GetKey() const{ return m_key; }


inline void SetKey(const Aws::String& value) { m_keyHasBeenSet = true; m_key = value; }


inline void SetKey(Aws::String&& value) { m_keyHasBeenSet = true; m_key = std::move(value); }


inline void SetKey(const char* value) { m_keyHasBeenSet = true; m_key.assign(value); }


inline MultipartUpload& WithKey(const Aws::String& value) { SetKey(value); return *this;}


inline MultipartUpload& WithKey(Aws::String&& value) { SetKey(std::move(value)); return *this;}


inline MultipartUpload& WithKey(const char* value) { SetKey(value); return *this;}



inline const Aws::Utils::DateTime& GetInitiated() const{ return m_initiated; }


inline void SetInitiated(const Aws::Utils::DateTime& value) { m_initiatedHasBeenSet = true; m_initiated = value; }


inline void SetInitiated(Aws::Utils::DateTime&& value) { m_initiatedHasBeenSet = true; m_initiated = std::move(value); }


inline MultipartUpload& WithInitiated(const Aws::Utils::DateTime& value) { SetInitiated(value); return *this;}


inline MultipartUpload& WithInitiated(Aws::Utils::DateTime&& value) { SetInitiated(std::move(value)); return *this;}



inline const StorageClass& GetStorageClass() const{ return m_storageClass; }


inline void SetStorageClass(const StorageClass& value) { m_storageClassHasBeenSet = true; m_storageClass = value; }


inline void SetStorageClass(StorageClass&& value) { m_storageClassHasBeenSet = true; m_storageClass = std::move(value); }


inline MultipartUpload& WithStorageClass(const StorageClass& value) { SetStorageClass(value); return *this;}


inline MultipartUpload& WithStorageClass(StorageClass&& value) { SetStorageClass(std::move(value)); return *this;}



inline const Owner& GetOwner() const{ return m_owner; }


inline void SetOwner(const Owner& value) { m_ownerHasBeenSet = true; m_owner = value; }


inline void SetOwner(Owner&& value) { m_ownerHasBeenSet = true; m_owner = std::move(value); }


inline MultipartUpload& WithOwner(const Owner& value) { SetOwner(value); return *this;}


inline MultipartUpload& WithOwner(Owner&& value) { SetOwner(std::move(value)); return *this;}



inline const Initiator& GetInitiator() const{ return m_initiator; }


inline void SetInitiator(const Initiator& value) { m_initiatorHasBeenSet = true; m_initiator = value; }


inline void SetInitiator(Initiator&& value) { m_initiatorHasBeenSet = true; m_initiator = std::move(value); }


inline MultipartUpload& WithInitiator(const Initiator& value) { SetInitiator(value); return *this;}


inline MultipartUpload& WithInitiator(Initiator&& value) { SetInitiator(std::move(value)); return *this;}

private:

Aws::String m_uploadId;
bool m_uploadIdHasBeenSet;

Aws::String m_key;
bool m_keyHasBeenSet;

Aws::Utils::DateTime m_initiated;
bool m_initiatedHasBeenSet;

StorageClass m_storageClass;
bool m_storageClassHasBeenSet;

Owner m_owner;
bool m_ownerHasBeenSet;

Initiator m_initiator;
bool m_initiatorHasBeenSet;
};

} 
} 
} 
