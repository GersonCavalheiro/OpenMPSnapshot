

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSStreamFwd.h>
#include <aws/core/utils/UnreferencedParam.h>
#include <aws/core/http/HttpTypes.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/AmazonWebServiceRequest.h>

namespace Aws
{
static const char* DEFAULT_CONTENT_TYPE = "binary/octet-stream";


class AWS_CORE_API AmazonStreamingWebServiceRequest : public AmazonWebServiceRequest
{
public:

AmazonStreamingWebServiceRequest() : m_contentType(DEFAULT_CONTENT_TYPE)
{
}

virtual ~AmazonStreamingWebServiceRequest();


inline std::shared_ptr<Aws::IOStream> GetBody() const override { return m_bodyStream; }

inline void SetBody(const std::shared_ptr<Aws::IOStream>& body) { m_bodyStream = body; }

inline Aws::Http::HeaderValueCollection GetHeaders() const override
{
auto headers = GetRequestSpecificHeaders();
headers.insert(Aws::Http::HeaderValuePair(Aws::Http::CONTENT_TYPE_HEADER, GetContentType()));

return headers;
}


const Aws::String& GetContentType() const { return m_contentType; }

void SetContentType(const Aws::String& contentType) { m_contentType = contentType; }

protected:

virtual Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const { return Aws::Http::HeaderValueCollection(); };

private:
std::shared_ptr<Aws::IOStream> m_bodyStream;
Aws::String m_contentType;
};

} 

