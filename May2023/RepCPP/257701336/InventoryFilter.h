

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

class AWS_S3_API InventoryFilter
{
public:
InventoryFilter();
InventoryFilter(const Aws::Utils::Xml::XmlNode& xmlNode);
InventoryFilter& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetPrefix() const{ return m_prefix; }


inline void SetPrefix(const Aws::String& value) { m_prefixHasBeenSet = true; m_prefix = value; }


inline void SetPrefix(Aws::String&& value) { m_prefixHasBeenSet = true; m_prefix = std::move(value); }


inline void SetPrefix(const char* value) { m_prefixHasBeenSet = true; m_prefix.assign(value); }


inline InventoryFilter& WithPrefix(const Aws::String& value) { SetPrefix(value); return *this;}


inline InventoryFilter& WithPrefix(Aws::String&& value) { SetPrefix(std::move(value)); return *this;}


inline InventoryFilter& WithPrefix(const char* value) { SetPrefix(value); return *this;}

private:

Aws::String m_prefix;
bool m_prefixHasBeenSet;
};

} 
} 
} 
