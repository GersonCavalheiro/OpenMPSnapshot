

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

class AWS_S3_API IndexDocument
{
public:
IndexDocument();
IndexDocument(const Aws::Utils::Xml::XmlNode& xmlNode);
IndexDocument& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetSuffix() const{ return m_suffix; }


inline void SetSuffix(const Aws::String& value) { m_suffixHasBeenSet = true; m_suffix = value; }


inline void SetSuffix(Aws::String&& value) { m_suffixHasBeenSet = true; m_suffix = std::move(value); }


inline void SetSuffix(const char* value) { m_suffixHasBeenSet = true; m_suffix.assign(value); }


inline IndexDocument& WithSuffix(const Aws::String& value) { SetSuffix(value); return *this;}


inline IndexDocument& WithSuffix(Aws::String&& value) { SetSuffix(std::move(value)); return *this;}


inline IndexDocument& WithSuffix(const char* value) { SetSuffix(value); return *this;}

private:

Aws::String m_suffix;
bool m_suffixHasBeenSet;
};

} 
} 
} 
