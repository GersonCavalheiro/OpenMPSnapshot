

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/MetricsConfiguration.h>
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
class AWS_S3_API GetBucketMetricsConfigurationResult
{
public:
GetBucketMetricsConfigurationResult();
GetBucketMetricsConfigurationResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
GetBucketMetricsConfigurationResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const MetricsConfiguration& GetMetricsConfiguration() const{ return m_metricsConfiguration; }


inline void SetMetricsConfiguration(const MetricsConfiguration& value) { m_metricsConfiguration = value; }


inline void SetMetricsConfiguration(MetricsConfiguration&& value) { m_metricsConfiguration = std::move(value); }


inline GetBucketMetricsConfigurationResult& WithMetricsConfiguration(const MetricsConfiguration& value) { SetMetricsConfiguration(value); return *this;}


inline GetBucketMetricsConfigurationResult& WithMetricsConfiguration(MetricsConfiguration&& value) { SetMetricsConfiguration(std::move(value)); return *this;}

private:

MetricsConfiguration m_metricsConfiguration;
};

} 
} 
} 
