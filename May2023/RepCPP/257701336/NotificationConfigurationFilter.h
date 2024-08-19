

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/S3KeyFilter.h>
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


class AWS_S3_API NotificationConfigurationFilter
{
public:
NotificationConfigurationFilter();
NotificationConfigurationFilter(const Aws::Utils::Xml::XmlNode& xmlNode);
NotificationConfigurationFilter& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const S3KeyFilter& GetKey() const{ return m_key; }


inline void SetKey(const S3KeyFilter& value) { m_keyHasBeenSet = true; m_key = value; }


inline void SetKey(S3KeyFilter&& value) { m_keyHasBeenSet = true; m_key = std::move(value); }


inline NotificationConfigurationFilter& WithKey(const S3KeyFilter& value) { SetKey(value); return *this;}


inline NotificationConfigurationFilter& WithKey(S3KeyFilter&& value) { SetKey(std::move(value)); return *this;}

private:

S3KeyFilter m_key;
bool m_keyHasBeenSet;
};

} 
} 
} 
