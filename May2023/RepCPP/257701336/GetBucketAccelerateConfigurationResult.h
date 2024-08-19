

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/BucketAccelerateStatus.h>
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
class AWS_S3_API GetBucketAccelerateConfigurationResult
{
public:
GetBucketAccelerateConfigurationResult();
GetBucketAccelerateConfigurationResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
GetBucketAccelerateConfigurationResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const BucketAccelerateStatus& GetStatus() const{ return m_status; }


inline void SetStatus(const BucketAccelerateStatus& value) { m_status = value; }


inline void SetStatus(BucketAccelerateStatus&& value) { m_status = std::move(value); }


inline GetBucketAccelerateConfigurationResult& WithStatus(const BucketAccelerateStatus& value) { SetStatus(value); return *this;}


inline GetBucketAccelerateConfigurationResult& WithStatus(BucketAccelerateStatus&& value) { SetStatus(std::move(value)); return *this;}

private:

BucketAccelerateStatus m_status;
};

} 
} 
} 
