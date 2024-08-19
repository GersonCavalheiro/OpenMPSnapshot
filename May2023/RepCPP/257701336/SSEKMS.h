

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


class AWS_S3_API SSEKMS
{
public:
SSEKMS();
SSEKMS(const Aws::Utils::Xml::XmlNode& xmlNode);
SSEKMS& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetKeyId() const{ return m_keyId; }


inline void SetKeyId(const Aws::String& value) { m_keyIdHasBeenSet = true; m_keyId = value; }


inline void SetKeyId(Aws::String&& value) { m_keyIdHasBeenSet = true; m_keyId = std::move(value); }


inline void SetKeyId(const char* value) { m_keyIdHasBeenSet = true; m_keyId.assign(value); }


inline SSEKMS& WithKeyId(const Aws::String& value) { SetKeyId(value); return *this;}


inline SSEKMS& WithKeyId(Aws::String&& value) { SetKeyId(std::move(value)); return *this;}


inline SSEKMS& WithKeyId(const char* value) { SetKeyId(value); return *this;}

private:

Aws::String m_keyId;
bool m_keyIdHasBeenSet;
};

} 
} 
} 
