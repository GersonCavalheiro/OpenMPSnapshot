

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/AnalyticsConfiguration.h>
#include <utility>

namespace Aws
{
template<typename RESULT_TYPE>
class AmazonWebServiceResult;

namespace Utils
{
namespace Xml
{
class XmlDocument;
} 
} 
namespace S3
{
namespace Model
{
class AWS_S3_API GetBucketAnalyticsConfigurationResult
{
public:
GetBucketAnalyticsConfigurationResult();
GetBucketAnalyticsConfigurationResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
GetBucketAnalyticsConfigurationResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const AnalyticsConfiguration& GetAnalyticsConfiguration() const{ return m_analyticsConfiguration; }


inline void SetAnalyticsConfiguration(const AnalyticsConfiguration& value) { m_analyticsConfiguration = value; }


inline void SetAnalyticsConfiguration(AnalyticsConfiguration&& value) { m_analyticsConfiguration = std::move(value); }


inline GetBucketAnalyticsConfigurationResult& WithAnalyticsConfiguration(const AnalyticsConfiguration& value) { SetAnalyticsConfiguration(value); return *this;}


inline GetBucketAnalyticsConfigurationResult& WithAnalyticsConfiguration(AnalyticsConfiguration&& value) { SetAnalyticsConfiguration(std::move(value)); return *this;}

private:

AnalyticsConfiguration m_analyticsConfiguration;
};

} 
} 
} 
