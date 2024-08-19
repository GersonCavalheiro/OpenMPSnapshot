

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/ObjectVersionStorageClass.h>
#include <aws/core/utils/DateTime.h>
#include <aws/s3/model/Owner.h>
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

class AWS_S3_API ObjectVersion
{
public:
ObjectVersion();
ObjectVersion(const Aws::Utils::Xml::XmlNode& xmlNode);
ObjectVersion& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetETag() const{ return m_eTag; }


inline void SetETag(const Aws::String& value) { m_eTagHasBeenSet = true; m_eTag = value; }


inline void SetETag(Aws::String&& value) { m_eTagHasBeenSet = true; m_eTag = std::move(value); }


inline void SetETag(const char* value) { m_eTagHasBeenSet = true; m_eTag.assign(value); }


inline ObjectVersion& WithETag(const Aws::String& value) { SetETag(value); return *this;}


inline ObjectVersion& WithETag(Aws::String&& value) { SetETag(std::move(value)); return *this;}


inline ObjectVersion& WithETag(const char* value) { SetETag(value); return *this;}



inline long long GetSize() const{ return m_size; }


inline void SetSize(long long value) { m_sizeHasBeenSet = true; m_size = value; }


inline ObjectVersion& WithSize(long long value) { SetSize(value); return *this;}



inline const ObjectVersionStorageClass& GetStorageClass() const{ return m_storageClass; }


inline void SetStorageClass(const ObjectVersionStorageClass& value) { m_storageClassHasBeenSet = true; m_storageClass = value; }


inline void SetStorageClass(ObjectVersionStorageClass&& value) { m_storageClassHasBeenSet = true; m_storageClass = std::move(value); }


inline ObjectVersion& WithStorageClass(const ObjectVersionStorageClass& value) { SetStorageClass(value); return *this;}


inline ObjectVersion& WithStorageClass(ObjectVersionStorageClass&& value) { SetStorageClass(std::move(value)); return *this;}



inline const Aws::String& GetKey() const{ return m_key; }


inline void SetKey(const Aws::String& value) { m_keyHasBeenSet = true; m_key = value; }


inline void SetKey(Aws::String&& value) { m_keyHasBeenSet = true; m_key = std::move(value); }


inline void SetKey(const char* value) { m_keyHasBeenSet = true; m_key.assign(value); }


inline ObjectVersion& WithKey(const Aws::String& value) { SetKey(value); return *this;}


inline ObjectVersion& WithKey(Aws::String&& value) { SetKey(std::move(value)); return *this;}


inline ObjectVersion& WithKey(const char* value) { SetKey(value); return *this;}



inline const Aws::String& GetVersionId() const{ return m_versionId; }


inline void SetVersionId(const Aws::String& value) { m_versionIdHasBeenSet = true; m_versionId = value; }


inline void SetVersionId(Aws::String&& value) { m_versionIdHasBeenSet = true; m_versionId = std::move(value); }


inline void SetVersionId(const char* value) { m_versionIdHasBeenSet = true; m_versionId.assign(value); }


inline ObjectVersion& WithVersionId(const Aws::String& value) { SetVersionId(value); return *this;}


inline ObjectVersion& WithVersionId(Aws::String&& value) { SetVersionId(std::move(value)); return *this;}


inline ObjectVersion& WithVersionId(const char* value) { SetVersionId(value); return *this;}



inline bool GetIsLatest() const{ return m_isLatest; }


inline void SetIsLatest(bool value) { m_isLatestHasBeenSet = true; m_isLatest = value; }


inline ObjectVersion& WithIsLatest(bool value) { SetIsLatest(value); return *this;}



inline const Aws::Utils::DateTime& GetLastModified() const{ return m_lastModified; }


inline void SetLastModified(const Aws::Utils::DateTime& value) { m_lastModifiedHasBeenSet = true; m_lastModified = value; }


inline void SetLastModified(Aws::Utils::DateTime&& value) { m_lastModifiedHasBeenSet = true; m_lastModified = std::move(value); }


inline ObjectVersion& WithLastModified(const Aws::Utils::DateTime& value) { SetLastModified(value); return *this;}


inline ObjectVersion& WithLastModified(Aws::Utils::DateTime&& value) { SetLastModified(std::move(value)); return *this;}



inline const Owner& GetOwner() const{ return m_owner; }


inline void SetOwner(const Owner& value) { m_ownerHasBeenSet = true; m_owner = value; }


inline void SetOwner(Owner&& value) { m_ownerHasBeenSet = true; m_owner = std::move(value); }


inline ObjectVersion& WithOwner(const Owner& value) { SetOwner(value); return *this;}


inline ObjectVersion& WithOwner(Owner&& value) { SetOwner(std::move(value)); return *this;}

private:

Aws::String m_eTag;
bool m_eTagHasBeenSet;

long long m_size;
bool m_sizeHasBeenSet;

ObjectVersionStorageClass m_storageClass;
bool m_storageClassHasBeenSet;

Aws::String m_key;
bool m_keyHasBeenSet;

Aws::String m_versionId;
bool m_versionIdHasBeenSet;

bool m_isLatest;
bool m_isLatestHasBeenSet;

Aws::Utils::DateTime m_lastModified;
bool m_lastModifiedHasBeenSet;

Owner m_owner;
bool m_ownerHasBeenSet;
};

} 
} 
} 
