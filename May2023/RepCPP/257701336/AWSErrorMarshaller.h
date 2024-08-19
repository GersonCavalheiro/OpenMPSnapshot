

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace Http
{
class HttpResponse;
enum class HttpResponseCode;
}

namespace Client
{
enum class CoreErrors;

template<typename ERROR_TYPE>
class AWSError;


class AWS_CORE_API AWSErrorMarshaller
{
public:            
virtual ~AWSErrorMarshaller() {}


virtual AWSError<CoreErrors> Marshall(const Aws::Http::HttpResponse& response) const = 0;

virtual AWSError<CoreErrors> FindErrorByName(const char* exceptionName) const;
virtual AWSError<CoreErrors> FindErrorByHttpResponseCode(Aws::Http::HttpResponseCode code) const;
protected:
AWSError<CoreErrors> Marshall(const Aws::String& exceptionName, const Aws::String& message) const;
};

class AWS_CORE_API JsonErrorMarshaller : public AWSErrorMarshaller
{
using AWSErrorMarshaller::Marshall;
public:

AWSError<CoreErrors> Marshall(const Aws::Http::HttpResponse& response) const override;
};

class AWS_CORE_API XmlErrorMarshaller : public AWSErrorMarshaller
{
using AWSErrorMarshaller::Marshall;
public:

AWSError<CoreErrors> Marshall(const Aws::Http::HttpResponse& response) const override;
};

} 
} 
