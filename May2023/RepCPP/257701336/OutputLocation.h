

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/S3Location.h>
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


class AWS_S3_API OutputLocation
{
public:
OutputLocation();
OutputLocation(const Aws::Utils::Xml::XmlNode& xmlNode);
OutputLocation& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const S3Location& GetS3() const{ return m_s3; }


inline void SetS3(const S3Location& value) { m_s3HasBeenSet = true; m_s3 = value; }


inline void SetS3(S3Location&& value) { m_s3HasBeenSet = true; m_s3 = std::move(value); }


inline OutputLocation& WithS3(const S3Location& value) { SetS3(value); return *this;}


inline OutputLocation& WithS3(S3Location&& value) { SetS3(std::move(value)); return *this;}

private:

S3Location m_s3;
bool m_s3HasBeenSet;
};

} 
} 
} 
