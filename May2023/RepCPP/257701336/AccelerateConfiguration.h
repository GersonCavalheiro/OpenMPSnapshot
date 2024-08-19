

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/BucketAccelerateStatus.h>
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

class AWS_S3_API AccelerateConfiguration
{
public:
AccelerateConfiguration();
AccelerateConfiguration(const Aws::Utils::Xml::XmlNode& xmlNode);
AccelerateConfiguration& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const BucketAccelerateStatus& GetStatus() const{ return m_status; }


inline void SetStatus(const BucketAccelerateStatus& value) { m_statusHasBeenSet = true; m_status = value; }


inline void SetStatus(BucketAccelerateStatus&& value) { m_statusHasBeenSet = true; m_status = std::move(value); }


inline AccelerateConfiguration& WithStatus(const BucketAccelerateStatus& value) { SetStatus(value); return *this;}


inline AccelerateConfiguration& WithStatus(BucketAccelerateStatus&& value) { SetStatus(std::move(value)); return *this;}

private:

BucketAccelerateStatus m_status;
bool m_statusHasBeenSet;
};

} 
} 
} 
