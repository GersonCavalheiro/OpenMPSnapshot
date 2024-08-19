

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
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
class AWS_S3_API CreateBucketResult
{
public:
CreateBucketResult();
CreateBucketResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
CreateBucketResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const Aws::String& GetLocation() const{ return m_location; }


inline void SetLocation(const Aws::String& value) { m_location = value; }


inline void SetLocation(Aws::String&& value) { m_location = std::move(value); }


inline void SetLocation(const char* value) { m_location.assign(value); }


inline CreateBucketResult& WithLocation(const Aws::String& value) { SetLocation(value); return *this;}


inline CreateBucketResult& WithLocation(Aws::String&& value) { SetLocation(std::move(value)); return *this;}


inline CreateBucketResult& WithLocation(const char* value) { SetLocation(value); return *this;}

private:

Aws::String m_location;
};

} 
} 
} 
