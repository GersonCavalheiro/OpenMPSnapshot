

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/http/URI.h>
#include <aws/core/http/HttpTypes.h>
#include <aws/core/utils/memory/AWSMemory.h>
#include <aws/core/utils/memory/stl/AWSStreamFwd.h>
#include <aws/core/utils/stream/ResponseStream.h>
#include <memory>
#include <functional>

namespace Aws
{
namespace Http
{
extern AWS_CORE_API const char* DATE_HEADER;
extern AWS_CORE_API const char* AWS_DATE_HEADER;
extern AWS_CORE_API const char* AWS_SECURITY_TOKEN;
extern AWS_CORE_API const char* ACCEPT_HEADER;
extern AWS_CORE_API const char* ACCEPT_CHAR_SET_HEADER;
extern AWS_CORE_API const char* ACCEPT_ENCODING_HEADER;
extern AWS_CORE_API const char* AUTHORIZATION_HEADER;
extern AWS_CORE_API const char* AWS_AUTHORIZATION_HEADER;
extern AWS_CORE_API const char* COOKIE_HEADER;
extern AWS_CORE_API const char* CONTENT_LENGTH_HEADER;
extern AWS_CORE_API const char* CONTENT_TYPE_HEADER;
extern AWS_CORE_API const char* USER_AGENT_HEADER;
extern AWS_CORE_API const char* VIA_HEADER;
extern AWS_CORE_API const char* HOST_HEADER;
extern AWS_CORE_API const char* AMZ_TARGET_HEADER;
extern AWS_CORE_API const char* X_AMZ_EXPIRES_HEADER;
extern AWS_CORE_API const char* CONTENT_MD5_HEADER;

class HttpRequest;
class HttpResponse;


typedef std::function<void(const HttpRequest*, HttpResponse*, long long)> DataReceivedEventHandler;

typedef std::function<void(const HttpRequest*, long long)> DataSentEventHandler;

typedef std::function<bool(const HttpRequest*)> ContinueRequestHandler;


class AWS_CORE_API HttpRequest
{
public:

HttpRequest(const URI& uri, HttpMethod method) :
m_uri(uri), m_method(method)
{}

virtual ~HttpRequest() {}


virtual HeaderValueCollection GetHeaders() const = 0;

virtual const Aws::String& GetHeaderValue(const char* headerName) const = 0;

virtual void SetHeaderValue(const char* headerName, const Aws::String& headerValue) = 0;

virtual void SetHeaderValue(const Aws::String& headerName, const Aws::String& headerValue) = 0;

virtual void DeleteHeader(const char* headerName) = 0;

virtual void AddContentBody(const std::shared_ptr<Aws::IOStream>& strContent) = 0;

virtual const std::shared_ptr<Aws::IOStream>& GetContentBody() const = 0;

virtual bool HasHeader(const char* name) const = 0;

virtual int64_t GetSize() const = 0;

virtual const Aws::IOStreamFactory& GetResponseStreamFactory() const = 0;

virtual void SetResponseStreamFactory(const Aws::IOStreamFactory& factory) = 0;

inline URI& GetUri()
{
return m_uri;
}

const URI& GetUri() const { return m_uri; }

inline Aws::String GetURIString(bool includeQueryString = true) const
{
return m_uri.GetURIString(includeQueryString);
}

inline HttpMethod GetMethod() const
{
return m_method;
}

inline const Aws::String& GetQueryString() const
{
return m_uri.GetQueryString();
}

inline void CanonicalizeRequest()
{
m_uri.CanonicalizeQueryString();
}

inline QueryStringParameterCollection GetQueryStringParameters() const
{
return m_uri.GetQueryStringParameters();
}

inline void AddQueryStringParameter(const char* key, const Aws::String& value)
{
m_uri.AddQueryStringParameter(key, value);
}


inline const Aws::String& GetDate() const
{
return GetHeaderValue(DATE_HEADER);
}

inline void SetDate(const Aws::String& value)
{
SetHeaderValue(DATE_HEADER, value);
}

inline const Aws::String& GetAccept() const
{
return GetHeaderValue(ACCEPT_HEADER);
}

inline void SetAccept(const Aws::String& value)
{
SetHeaderValue(ACCEPT_HEADER, value);
}

inline const Aws::String& GetAcceptCharSet() const
{
return GetHeaderValue(ACCEPT_CHAR_SET_HEADER);
}

inline void SetAcceptCharSet(const Aws::String& value)
{
SetHeaderValue(ACCEPT_CHAR_SET_HEADER, value);
}

inline const Aws::String& GetAcceptEncoding() const
{
return GetHeaderValue(ACCEPT_ENCODING_HEADER);
}

inline void SetAcceptEncoding(const Aws::String& value)
{
SetHeaderValue(ACCEPT_ENCODING_HEADER, value);
}

inline const Aws::String& GetAuthorization() const
{
return GetHeaderValue(AUTHORIZATION_HEADER);
}

inline void SetAuthorization(const Aws::String& value)
{
SetHeaderValue(AUTHORIZATION_HEADER, value);
}

inline const Aws::String& GetAwsAuthorization() const
{
return GetHeaderValue(AWS_AUTHORIZATION_HEADER);
}

inline void SetAwsAuthorization(const Aws::String& value)
{
SetHeaderValue(AWS_AUTHORIZATION_HEADER, value);
}

inline const Aws::String& GetAwsSessionToken() const
{
return GetHeaderValue(AWS_SECURITY_TOKEN);
}

inline void SetAwsSessionToken(const Aws::String& value)
{
SetHeaderValue(AWS_SECURITY_TOKEN, value);
}

inline const Aws::String& GetCookie() const
{
return GetHeaderValue(COOKIE_HEADER);
}

inline void SetCookie(const Aws::String& value)
{
SetHeaderValue(COOKIE_HEADER, value);
}

inline const Aws::String& GetContentLength() const
{
return GetHeaderValue(CONTENT_LENGTH_HEADER);
}

inline void SetContentLength(const Aws::String& value)
{
SetHeaderValue(CONTENT_LENGTH_HEADER, value);
}

inline const Aws::String& GetContentType() const
{
return GetHeaderValue(CONTENT_TYPE_HEADER);
}

inline void SetContentType(const Aws::String& value)
{
SetHeaderValue(CONTENT_TYPE_HEADER, value);
}

inline const Aws::String& GetUserAgent() const
{
return GetHeaderValue(USER_AGENT_HEADER);
}

inline void SetUserAgent(const Aws::String& value)
{
SetHeaderValue(USER_AGENT_HEADER, value);
}

inline const Aws::String& GetVia() const
{
return GetHeaderValue(VIA_HEADER);
}

inline void SetVia(const Aws::String& value)
{
SetHeaderValue(VIA_HEADER, value);
}

inline void SetDataReceivedEventHandler(const DataReceivedEventHandler& dataReceivedEventHandler) { m_onDataReceived = dataReceivedEventHandler; }

inline void SetDataReceivedEventHandler(DataReceivedEventHandler&& dataReceivedEventHandler) { m_onDataReceived = std::move(dataReceivedEventHandler); }

inline void SetDataSentEventHandler(const DataSentEventHandler& dataSentEventHandler) { m_onDataSent = dataSentEventHandler; }

inline void SetDataSentEventHandler(DataSentEventHandler&& dataSentEventHandler) { m_onDataSent = std::move(dataSentEventHandler); }

inline void SetContinueRequestHandle(const ContinueRequestHandler& continueRequestHandler) { m_continueRequest = continueRequestHandler; }

inline void SetContinueRequestHandle(ContinueRequestHandler&& continueRequestHandler) { m_continueRequest = std::move(continueRequestHandler); }


inline const DataReceivedEventHandler& GetDataReceivedEventHandler() const { return m_onDataReceived; }

inline const DataSentEventHandler& GetDataSentEventHandler() const { return m_onDataSent; }

inline const ContinueRequestHandler& GetContinueRequestHandler() const { return m_continueRequest; }
private:
URI m_uri;
HttpMethod m_method;
DataReceivedEventHandler m_onDataReceived;
DataSentEventHandler m_onDataSent;
ContinueRequestHandler m_continueRequest;
};

} 
} 



