

#pragma once
#include <aws/s3/S3_EXPORTS.h>

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


class AWS_S3_API NoncurrentVersionExpiration
{
public:
NoncurrentVersionExpiration();
NoncurrentVersionExpiration(const Aws::Utils::Xml::XmlNode& xmlNode);
NoncurrentVersionExpiration& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline int GetNoncurrentDays() const{ return m_noncurrentDays; }


inline void SetNoncurrentDays(int value) { m_noncurrentDaysHasBeenSet = true; m_noncurrentDays = value; }


inline NoncurrentVersionExpiration& WithNoncurrentDays(int value) { SetNoncurrentDays(value); return *this;}

private:

int m_noncurrentDays;
bool m_noncurrentDaysHasBeenSet;
};

} 
} 
} 
