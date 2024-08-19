

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/memory/stl/AWSFunction.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/UnreferencedParam.h>
#include <aws/core/http/HttpTypes.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/utils/memory/stl/AWSStreamFwd.h>
#include <aws/core/utils/stream/ResponseStream.h>
#include <aws/core/auth/AWSAuthSigner.h>

namespace Aws
{
namespace Http
{
class URI;
} 

class AmazonWebServiceRequest;


typedef std::function<void(const AmazonWebServiceRequest&)> RequestRetryHandler;


class AWS_CORE_API AmazonWebServiceRequest
{
public:

AmazonWebServiceRequest();
virtual ~AmazonWebServiceRequest() = default;


virtual std::shared_ptr<Aws::IOStream> GetBody() const = 0;

virtual Aws::Http::HeaderValueCollection GetHeaders() const = 0;

virtual void AddQueryStringParameters(Aws::Http::URI& uri) const { AWS_UNREFERENCED_PARAM(uri); }


virtual void PutToPresignedUrl(Aws::Http::URI& uri) const { DumpBodyToUrl(uri); AddQueryStringParameters(uri); }


virtual bool SignBody() const { return true; }


const Aws::IOStreamFactory& GetResponseStreamFactory() const { return m_responseStreamFactory; }

void SetResponseStreamFactory(const Aws::IOStreamFactory& factory) { m_responseStreamFactory = AWS_BUILD_FUNCTION(factory); }

inline virtual void SetDataReceivedEventHandler(const Aws::Http::DataReceivedEventHandler& dataReceivedEventHandler) { m_onDataReceived = dataReceivedEventHandler; }

inline virtual void SetDataSentEventHandler(const Aws::Http::DataSentEventHandler& dataSentEventHandler) { m_onDataSent = dataSentEventHandler; }

inline virtual void SetContinueRequestHandler(const Aws::Http::ContinueRequestHandler& continueRequestHandler) { m_continueRequest = continueRequestHandler; }

inline virtual void SetDataReceivedEventHandler(Aws::Http::DataReceivedEventHandler&& dataReceivedEventHandler) { m_onDataReceived = std::move(dataReceivedEventHandler); }

inline virtual void SetDataSentEventHandler(Aws::Http::DataSentEventHandler&& dataSentEventHandler) { m_onDataSent = std::move(dataSentEventHandler); }

inline virtual void SetContinueRequestHandler(Aws::Http::ContinueRequestHandler&& continueRequestHandler) { m_continueRequest = std::move(continueRequestHandler); }

inline virtual void SetRequestRetryHandler(const RequestRetryHandler& handler) { m_requestRetryHandler = handler; }

inline virtual void SetRequestRetryHandler(RequestRetryHandler&& handler) { m_requestRetryHandler = std::move(handler); }

inline virtual const Aws::Http::DataReceivedEventHandler& GetDataReceivedEventHandler() const { return m_onDataReceived; }

inline virtual const Aws::Http::DataSentEventHandler& GetDataSentEventHandler() const { return m_onDataSent; }

inline virtual const Aws::Http::ContinueRequestHandler& GetContinueRequestHandler() const { return m_continueRequest; }

inline virtual const RequestRetryHandler& GetRequestRetryHandler() const { return m_requestRetryHandler; }

inline virtual bool ShouldComputeContentMd5() const { return false; }

virtual const char* GetServiceRequestName() const = 0;

protected:

virtual void DumpBodyToUrl(Aws::Http::URI& uri) const { AWS_UNREFERENCED_PARAM(uri); }

private:
Aws::IOStreamFactory m_responseStreamFactory;

Aws::Http::DataReceivedEventHandler m_onDataReceived;
Aws::Http::DataSentEventHandler m_onDataSent;
Aws::Http::ContinueRequestHandler m_continueRequest;
RequestRetryHandler m_requestRetryHandler;
};

} 

