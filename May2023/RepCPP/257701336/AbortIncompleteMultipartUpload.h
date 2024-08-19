

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


class AWS_S3_API AbortIncompleteMultipartUpload
{
public:
AbortIncompleteMultipartUpload();
AbortIncompleteMultipartUpload(const Aws::Utils::Xml::XmlNode& xmlNode);
AbortIncompleteMultipartUpload& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline int GetDaysAfterInitiation() const{ return m_daysAfterInitiation; }


inline void SetDaysAfterInitiation(int value) { m_daysAfterInitiationHasBeenSet = true; m_daysAfterInitiation = value; }


inline AbortIncompleteMultipartUpload& WithDaysAfterInitiation(int value) { SetDaysAfterInitiation(value); return *this;}

private:

int m_daysAfterInitiation;
bool m_daysAfterInitiationHasBeenSet;
};

} 
} 
} 
