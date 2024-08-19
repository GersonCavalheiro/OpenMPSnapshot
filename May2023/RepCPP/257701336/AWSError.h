

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/utils/StringUtils.h>

namespace Aws
{
namespace Client
{
enum class CoreErrors;


template<typename ERROR_TYPE>
class AWSError
{
public:

AWSError() : m_isRetryable(false) {}

AWSError(ERROR_TYPE errorType, Aws::String exceptionName, const Aws::String message, bool isRetryable) :
m_errorType(errorType), m_exceptionName(exceptionName), m_message(message), m_isRetryable(isRetryable) {}

AWSError(ERROR_TYPE errorType, bool isRetryable) :
m_errorType(errorType), m_isRetryable(isRetryable) {}

AWSError(const AWSError<CoreErrors>& rhs) :
m_errorType(static_cast<ERROR_TYPE>(rhs.GetErrorType())), m_exceptionName(rhs.GetExceptionName()), 
m_message(rhs.GetMessage()), m_responseHeaders(rhs.GetResponseHeaders()), 
m_responseCode(rhs.GetResponseCode()), m_isRetryable(rhs.ShouldRetry())
{}          


inline const ERROR_TYPE GetErrorType() const { return m_errorType; }

inline const Aws::String& GetExceptionName() const { return m_exceptionName; }

inline void SetExceptionName(const Aws::String& exceptionName) { m_exceptionName = exceptionName; }

inline const Aws::String& GetMessage() const { return m_message; }

inline void SetMessage(const Aws::String& message) { m_message = message; }

inline bool ShouldRetry() const { return m_isRetryable; }

inline const Aws::Http::HeaderValueCollection& GetResponseHeaders() const { return m_responseHeaders; }

inline void SetResponseHeaders(const Aws::Http::HeaderValueCollection& headers) { m_responseHeaders = headers; }

inline bool ResponseHeaderExists(const Aws::String& headerName) const { return m_responseHeaders.find(Aws::Utils::StringUtils::ToLower(headerName.c_str())) != m_responseHeaders.end(); }

inline Aws::Http::HttpResponseCode GetResponseCode() const { return m_responseCode; }

inline void SetResponseCode(Aws::Http::HttpResponseCode responseCode) { m_responseCode = responseCode; }

private:
ERROR_TYPE m_errorType;
Aws::String m_exceptionName;
Aws::String m_message;
Aws::Http::HeaderValueCollection m_responseHeaders;
Aws::Http::HttpResponseCode m_responseCode;
bool m_isRetryable;
};

} 
} 
