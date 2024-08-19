

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/AmazonSerializableWebServiceRequest.h>
#include <aws/core/utils/UnreferencedParam.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/AmazonStreamingWebServiceRequest.h>

namespace Aws
{
namespace S3
{
class AWS_S3_API S3Request : public Aws::AmazonSerializableWebServiceRequest
{
public:
virtual ~S3Request () {}
virtual Aws::String SerializePayload() const override = 0;

void AddParametersToRequest(Aws::Http::HttpRequest& httpRequest) const { AWS_UNREFERENCED_PARAM(httpRequest); }

inline Aws::Http::HeaderValueCollection GetHeaders() const override
{
auto headers = GetRequestSpecificHeaders();

if(headers.size() == 0 || (headers.size() > 0 && headers.count(Aws::Http::CONTENT_TYPE_HEADER) == 0))
{
headers.insert(Aws::Http::HeaderValuePair(Aws::Http::CONTENT_TYPE_HEADER, Aws::AMZN_XML_CONTENT_TYPE ));
}

return headers;
}

protected:
virtual Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const { return Aws::Http::HeaderValueCollection(); }

};

typedef Aws::AmazonStreamingWebServiceRequest StreamingS3Request;

} 
} 
