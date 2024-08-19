

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

class AWS_S3_API ErrorDocument
{
public:
ErrorDocument();
ErrorDocument(const Aws::Utils::Xml::XmlNode& xmlNode);
ErrorDocument& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetKey() const{ return m_key; }


inline void SetKey(const Aws::String& value) { m_keyHasBeenSet = true; m_key = value; }


inline void SetKey(Aws::String&& value) { m_keyHasBeenSet = true; m_key = std::move(value); }


inline void SetKey(const char* value) { m_keyHasBeenSet = true; m_key.assign(value); }


inline ErrorDocument& WithKey(const Aws::String& value) { SetKey(value); return *this;}


inline ErrorDocument& WithKey(Aws::String&& value) { SetKey(std::move(value)); return *this;}


inline ErrorDocument& WithKey(const char* value) { SetKey(value); return *this;}

private:

Aws::String m_key;
bool m_keyHasBeenSet;
};

} 
} 
} 
