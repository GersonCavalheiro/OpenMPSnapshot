

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/BucketLocationConstraint.h>
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
class AWS_S3_API GetBucketLocationResult
{
public:
GetBucketLocationResult();
GetBucketLocationResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
GetBucketLocationResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const BucketLocationConstraint& GetLocationConstraint() const{ return m_locationConstraint; }


inline void SetLocationConstraint(const BucketLocationConstraint& value) { m_locationConstraint = value; }


inline void SetLocationConstraint(BucketLocationConstraint&& value) { m_locationConstraint = std::move(value); }


inline GetBucketLocationResult& WithLocationConstraint(const BucketLocationConstraint& value) { SetLocationConstraint(value); return *this;}


inline GetBucketLocationResult& WithLocationConstraint(BucketLocationConstraint&& value) { SetLocationConstraint(std::move(value)); return *this;}

private:

BucketLocationConstraint m_locationConstraint;
};

} 
} 
} 
