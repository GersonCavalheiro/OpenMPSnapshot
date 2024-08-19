

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/memory/stl/AWSList.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSMap.h>

#include <memory>

namespace Aws
{
namespace Http
{

enum class HttpMethod
{
HTTP_GET,
HTTP_POST,
HTTP_DELETE,
HTTP_PUT,
HTTP_HEAD,
HTTP_PATCH
};


enum class TransferLibType
{
DEFAULT_CLIENT,
CURL_CLIENT,
WIN_INET_CLIENT,
WIN_HTTP_CLIENT
};

namespace HttpMethodMapper
{

AWS_CORE_API const char* GetNameForHttpMethod(HttpMethod httpMethod);
} 

typedef std::pair<Aws::String, Aws::String> HeaderValuePair;
typedef Aws::Map<Aws::String, Aws::String> HeaderValueCollection;

} 
} 

