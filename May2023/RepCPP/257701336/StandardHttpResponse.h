

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/http/HttpResponse.h>
#include <aws/core/utils/memory/stl/AWSMap.h>
#include <aws/core/utils/stream/ResponseStream.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace Http
{
namespace Standard
{

class AWS_CORE_API StandardHttpResponse :
public HttpResponse
{
public:

StandardHttpResponse(const HttpRequest& originatingRequest) :
HttpResponse(originatingRequest),
headerMap(),
bodyStream(originatingRequest.GetResponseStreamFactory())
{}

~StandardHttpResponse() = default;


HeaderValueCollection GetHeaders() const;

bool HasHeader(const char* headerName) const;

const Aws::String& GetHeader(const Aws::String&) const;

inline Aws::IOStream& GetResponseBody() const { return bodyStream.GetUnderlyingStream(); }

inline Utils::Stream::ResponseStream&& SwapResponseStreamOwnership() { return std::move(bodyStream); }

void AddHeader(const Aws::String&, const Aws::String&);

private:
StandardHttpResponse(const StandardHttpResponse&);                

Aws::Map<Aws::String, Aws::String> headerMap;
Utils::Stream::ResponseStream bodyStream;
};

} 
} 
} 


