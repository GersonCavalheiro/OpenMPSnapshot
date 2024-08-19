

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/utils/memory/stl/AWSMap.h>
#include <aws/core/utils/memory/stl/AWSStreamFwd.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace Http
{
namespace Standard
{

class AWS_CORE_API StandardHttpRequest : public HttpRequest
{
public:

StandardHttpRequest(const URI& uri, HttpMethod method);


virtual HeaderValueCollection GetHeaders() const override;

virtual const Aws::String& GetHeaderValue(const char* headerName) const override;

virtual void SetHeaderValue(const char* headerName, const Aws::String& headerValue) override;

virtual void SetHeaderValue(const Aws::String& headerName, const Aws::String& headerValue) override;

virtual void DeleteHeader(const char* headerName) override;

virtual inline void AddContentBody(const std::shared_ptr<Aws::IOStream>& strContent) override { bodyStream = strContent; }

virtual inline const std::shared_ptr<Aws::IOStream>& GetContentBody() const override { return bodyStream; }

virtual bool HasHeader(const char*) const override;

virtual int64_t GetSize() const override;

virtual const Aws::IOStreamFactory& GetResponseStreamFactory() const override;

virtual void SetResponseStreamFactory(const Aws::IOStreamFactory& factory) override;

private:
HeaderValueCollection headerMap;
std::shared_ptr<Aws::IOStream> bodyStream;
Aws::IOStreamFactory m_responseStreamFactory;
};

} 
} 
} 


