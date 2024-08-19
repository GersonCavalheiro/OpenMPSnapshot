

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/CORSRule.h>
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
class AWS_S3_API GetBucketCorsResult
{
public:
GetBucketCorsResult();
GetBucketCorsResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
GetBucketCorsResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const Aws::Vector<CORSRule>& GetCORSRules() const{ return m_cORSRules; }


inline void SetCORSRules(const Aws::Vector<CORSRule>& value) { m_cORSRules = value; }


inline void SetCORSRules(Aws::Vector<CORSRule>&& value) { m_cORSRules = std::move(value); }


inline GetBucketCorsResult& WithCORSRules(const Aws::Vector<CORSRule>& value) { SetCORSRules(value); return *this;}


inline GetBucketCorsResult& WithCORSRules(Aws::Vector<CORSRule>&& value) { SetCORSRules(std::move(value)); return *this;}


inline GetBucketCorsResult& AddCORSRules(const CORSRule& value) { m_cORSRules.push_back(value); return *this; }


inline GetBucketCorsResult& AddCORSRules(CORSRule&& value) { m_cORSRules.push_back(std::move(value)); return *this; }

private:

Aws::Vector<CORSRule> m_cORSRules;
};

} 
} 
} 
