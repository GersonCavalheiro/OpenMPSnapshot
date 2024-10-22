

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/AnalyticsS3BucketDestination.h>
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

class AWS_S3_API AnalyticsExportDestination
{
public:
AnalyticsExportDestination();
AnalyticsExportDestination(const Aws::Utils::Xml::XmlNode& xmlNode);
AnalyticsExportDestination& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const AnalyticsS3BucketDestination& GetS3BucketDestination() const{ return m_s3BucketDestination; }


inline void SetS3BucketDestination(const AnalyticsS3BucketDestination& value) { m_s3BucketDestinationHasBeenSet = true; m_s3BucketDestination = value; }


inline void SetS3BucketDestination(AnalyticsS3BucketDestination&& value) { m_s3BucketDestinationHasBeenSet = true; m_s3BucketDestination = std::move(value); }


inline AnalyticsExportDestination& WithS3BucketDestination(const AnalyticsS3BucketDestination& value) { SetS3BucketDestination(value); return *this;}


inline AnalyticsExportDestination& WithS3BucketDestination(AnalyticsS3BucketDestination&& value) { SetS3BucketDestination(std::move(value)); return *this;}

private:

AnalyticsS3BucketDestination m_s3BucketDestination;
bool m_s3BucketDestinationHasBeenSet;
};

} 
} 
} 
