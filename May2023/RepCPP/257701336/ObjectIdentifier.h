

#pragma once
#include <aws/s3/S3_EXPORTS.h>
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

class AWS_S3_API ObjectIdentifier
{
public:
ObjectIdentifier();
ObjectIdentifier(const Aws::Utils::Xml::XmlNode& xmlNode);
ObjectIdentifier& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetKey() const{ return m_key; }


inline void SetKey(const Aws::String& value) { m_keyHasBeenSet = true; m_key = value; }


inline void SetKey(Aws::String&& value) { m_keyHasBeenSet = true; m_key = std::move(value); }


inline void SetKey(const char* value) { m_keyHasBeenSet = true; m_key.assign(value); }


inline ObjectIdentifier& WithKey(const Aws::String& value) { SetKey(value); return *this;}


inline ObjectIdentifier& WithKey(Aws::String&& value) { SetKey(std::move(value)); return *this;}


inline ObjectIdentifier& WithKey(const char* value) { SetKey(value); return *this;}



inline const Aws::String& GetVersionId() const{ return m_versionId; }


inline void SetVersionId(const Aws::String& value) { m_versionIdHasBeenSet = true; m_versionId = value; }


inline void SetVersionId(Aws::String&& value) { m_versionIdHasBeenSet = true; m_versionId = std::move(value); }


inline void SetVersionId(const char* value) { m_versionIdHasBeenSet = true; m_versionId.assign(value); }


inline ObjectIdentifier& WithVersionId(const Aws::String& value) { SetVersionId(value); return *this;}


inline ObjectIdentifier& WithVersionId(Aws::String&& value) { SetVersionId(std::move(value)); return *this;}


inline ObjectIdentifier& WithVersionId(const char* value) { SetVersionId(value); return *this;}

private:

Aws::String m_key;
bool m_keyHasBeenSet;

Aws::String m_versionId;
bool m_versionIdHasBeenSet;
};

} 
} 
} 
