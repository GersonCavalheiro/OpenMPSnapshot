

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/AmazonSerializableWebServiceRequest.h>
#include <aws/core/utils/UnreferencedParam.h>
#include <aws/core/http/HttpRequest.h>

namespace Aws
{
namespace Kinesis
{
class AWS_KINESIS_API KinesisRequest : public Aws::AmazonSerializableWebServiceRequest
{
public:
virtual ~KinesisRequest () {}
virtual Aws::String SerializePayload() const override = 0;

void AddParametersToRequest(Aws::Http::HttpRequest& httpRequest) const { AWS_UNREFERENCED_PARAM(httpRequest); }

inline Aws::Http::HeaderValueCollection GetHeaders() const override
{
auto headers = GetRequestSpecificHeaders();

if(headers.size() == 0 || (headers.size() > 0 && headers.count(Aws::Http::CONTENT_TYPE_HEADER) == 0))
{
headers.insert(Aws::Http::HeaderValuePair(Aws::Http::CONTENT_TYPE_HEADER, Aws::AMZN_JSON_CONTENT_TYPE_1_1 ));
}

return headers;
}

protected:
virtual Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const { return Aws::Http::HeaderValueCollection(); }

};


} 
} 
