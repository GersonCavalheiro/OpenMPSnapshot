

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/Payer.h>
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
class AWS_S3_API GetBucketRequestPaymentResult
{
public:
GetBucketRequestPaymentResult();
GetBucketRequestPaymentResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
GetBucketRequestPaymentResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const Payer& GetPayer() const{ return m_payer; }


inline void SetPayer(const Payer& value) { m_payer = value; }


inline void SetPayer(Payer&& value) { m_payer = std::move(value); }


inline GetBucketRequestPaymentResult& WithPayer(const Payer& value) { SetPayer(value); return *this;}


inline GetBucketRequestPaymentResult& WithPayer(Payer&& value) { SetPayer(std::move(value)); return *this;}

private:

Payer m_payer;
};

} 
} 
} 
