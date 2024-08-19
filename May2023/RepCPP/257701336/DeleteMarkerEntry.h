

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/Owner.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/DateTime.h>
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

class AWS_S3_API DeleteMarkerEntry
{
public:
DeleteMarkerEntry();
DeleteMarkerEntry(const Aws::Utils::Xml::XmlNode& xmlNode);
DeleteMarkerEntry& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Owner& GetOwner() const{ return m_owner; }


inline void SetOwner(const Owner& value) { m_ownerHasBeenSet = true; m_owner = value; }


inline void SetOwner(Owner&& value) { m_ownerHasBeenSet = true; m_owner = std::move(value); }


inline DeleteMarkerEntry& WithOwner(const Owner& value) { SetOwner(value); return *this;}


inline DeleteMarkerEntry& WithOwner(Owner&& value) { SetOwner(std::move(value)); return *this;}



inline const Aws::String& GetKey() const{ return m_key; }


inline void SetKey(const Aws::String& value) { m_keyHasBeenSet = true; m_key = value; }


inline void SetKey(Aws::String&& value) { m_keyHasBeenSet = true; m_key = std::move(value); }


inline void SetKey(const char* value) { m_keyHasBeenSet = true; m_key.assign(value); }


inline DeleteMarkerEntry& WithKey(const Aws::String& value) { SetKey(value); return *this;}


inline DeleteMarkerEntry& WithKey(Aws::String&& value) { SetKey(std::move(value)); return *this;}


inline DeleteMarkerEntry& WithKey(const char* value) { SetKey(value); return *this;}



inline const Aws::String& GetVersionId() const{ return m_versionId; }


inline void SetVersionId(const Aws::String& value) { m_versionIdHasBeenSet = true; m_versionId = value; }


inline void SetVersionId(Aws::String&& value) { m_versionIdHasBeenSet = true; m_versionId = std::move(value); }


inline void SetVersionId(const char* value) { m_versionIdHasBeenSet = true; m_versionId.assign(value); }


inline DeleteMarkerEntry& WithVersionId(const Aws::String& value) { SetVersionId(value); return *this;}


inline DeleteMarkerEntry& WithVersionId(Aws::String&& value) { SetVersionId(std::move(value)); return *this;}


inline DeleteMarkerEntry& WithVersionId(const char* value) { SetVersionId(value); return *this;}



inline bool GetIsLatest() const{ return m_isLatest; }


inline void SetIsLatest(bool value) { m_isLatestHasBeenSet = true; m_isLatest = value; }


inline DeleteMarkerEntry& WithIsLatest(bool value) { SetIsLatest(value); return *this;}



inline const Aws::Utils::DateTime& GetLastModified() const{ return m_lastModified; }


inline void SetLastModified(const Aws::Utils::DateTime& value) { m_lastModifiedHasBeenSet = true; m_lastModified = value; }


inline void SetLastModified(Aws::Utils::DateTime&& value) { m_lastModifiedHasBeenSet = true; m_lastModified = std::move(value); }


inline DeleteMarkerEntry& WithLastModified(const Aws::Utils::DateTime& value) { SetLastModified(value); return *this;}


inline DeleteMarkerEntry& WithLastModified(Aws::Utils::DateTime&& value) { SetLastModified(std::move(value)); return *this;}

private:

Owner m_owner;
bool m_ownerHasBeenSet;

Aws::String m_key;
bool m_keyHasBeenSet;

Aws::String m_versionId;
bool m_versionIdHasBeenSet;

bool m_isLatest;
bool m_isLatestHasBeenSet;

Aws::Utils::DateTime m_lastModified;
bool m_lastModifiedHasBeenSet;
};

} 
} 
} 
