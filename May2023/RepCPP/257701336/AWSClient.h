

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/client/CoreErrors.h>
#include <aws/core/http/HttpTypes.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/AmazonWebServiceResult.h>
#include <aws/core/utils/crypto/Hash.h>
#include <memory>
#include <atomic>

namespace Aws
{
namespace Utils
{
template<typename R, typename E>
class Outcome;

namespace Xml
{
class XmlDocument;
} 

namespace Json
{
class JsonValue;
} 

namespace RateLimits
{
class RateLimiterInterface;
} 

namespace Crypto
{
class MD5;
} 
} 

namespace Http
{
class HttpClient;

class HttpClientFactory;

class HttpRequest;

class HttpResponse;

class URI;
} 

namespace Auth
{
AWS_CORE_API extern const char SIGV4_SIGNER[];
AWS_CORE_API extern const char NULL_SIGNER[];
}

class AmazonWebServiceRequest;

namespace Client
{
template<typename ERROR_TYPE>
class AWSError;
class AWSErrorMarshaller;
class AWSRestfulJsonErrorMarshaller;
class AWSAuthSigner;
struct ClientConfiguration;
class RetryStrategy;

typedef Utils::Outcome<std::shared_ptr<Aws::Http::HttpResponse>, AWSError<CoreErrors>> HttpResponseOutcome;
typedef Utils::Outcome<AmazonWebServiceResult<Utils::Stream::ResponseStream>, AWSError<CoreErrors>> StreamOutcome;


class AWS_CORE_API AWSClient
{
public:

AWSClient(const Aws::Client::ClientConfiguration& configuration,
const std::shared_ptr<Aws::Client::AWSAuthSigner>& signer,
const std::shared_ptr<AWSErrorMarshaller>& errorMarshaller);


AWSClient(const Aws::Client::ClientConfiguration& configuration,
const Aws::Map<Aws::String, std::shared_ptr<Aws::Client::AWSAuthSigner>>& signerMap,
const std::shared_ptr<AWSErrorMarshaller>& errorMarshaller);

virtual ~AWSClient();


Aws::String GeneratePresignedUrl(Aws::Http::URI& uri, Aws::Http::HttpMethod method, long long expirationInSeconds = 0);


Aws::String GeneratePresignedUrl(Aws::Http::URI& uri, Aws::Http::HttpMethod method, const char* region, long long expirationInSeconds = 0) const;


Aws::String GeneratePresignedUrl(Aws::Http::URI& uri, Aws::Http::HttpMethod method, const char* region, const char* serviceName, long long expirationInSeconds = 0) const;

Aws::String GeneratePresignedUrl(const Aws::AmazonWebServiceRequest& request, Aws::Http::URI& uri, Aws::Http::HttpMethod method, 
const Aws::Http::QueryStringParameterCollection& extraParams = Aws::Http::QueryStringParameterCollection(), long long expirationInSeconds = 0) const;

Aws::String GeneratePresignedUrl(const Aws::AmazonWebServiceRequest& request, Aws::Http::URI& uri, Aws::Http::HttpMethod method, const char* region, const char* serviceName,
const Aws::Http::QueryStringParameterCollection& extraParams = Aws::Http::QueryStringParameterCollection(), long long expirationInSeconds = 0) const;

Aws::String GeneratePresignedUrl(const Aws::AmazonWebServiceRequest& request, Aws::Http::URI& uri, Aws::Http::HttpMethod method, const char* region,
const Aws::Http::QueryStringParameterCollection& extraParams = Aws::Http::QueryStringParameterCollection(), long long expirationInSeconds = 0) const;


void DisableRequestProcessing();


void EnableRequestProcessing();

inline virtual const char* GetServiceClientName() const { return nullptr; }

protected:

HttpResponseOutcome AttemptExhaustively(const Aws::Http::URI& uri,
const Aws::AmazonWebServiceRequest& request,
Http::HttpMethod httpMethod,
const char* signerName) const;


HttpResponseOutcome AttemptExhaustively(const Aws::Http::URI& uri, 
Http::HttpMethod httpMethod,
const char* signerName,
const char* requestName = nullptr) const;


HttpResponseOutcome AttemptOneRequest(const Aws::Http::URI& uri,
const Aws::AmazonWebServiceRequest& request,
Http::HttpMethod httpMethod,
const char* signerName) const;


HttpResponseOutcome AttemptOneRequest(const Aws::Http::URI& uri, 
Http::HttpMethod httpMethod,
const char* signerName,
const char* requestName = nullptr) const;


StreamOutcome MakeRequestWithUnparsedResponse(const Aws::Http::URI& uri,
const Aws::AmazonWebServiceRequest& request,
Http::HttpMethod method = Http::HttpMethod::HTTP_POST,
const char* signerName = Aws::Auth::SIGV4_SIGNER) const;


StreamOutcome MakeRequestWithUnparsedResponse(const Aws::Http::URI& uri,
Http::HttpMethod method = Http::HttpMethod::HTTP_POST,
const char* signerName = Aws::Auth::SIGV4_SIGNER,
const char* requestName = nullptr) const;


virtual AWSError<CoreErrors> BuildAWSError(const std::shared_ptr<Aws::Http::HttpResponse>& response) const = 0;


virtual void BuildHttpRequest(const Aws::AmazonWebServiceRequest& request,
const std::shared_ptr<Aws::Http::HttpRequest>& httpRequest) const;


const std::shared_ptr<AWSErrorMarshaller>& GetErrorMarshaller() const
{
return m_errorMarshaller;
}


Aws::Client::AWSAuthSigner* GetSignerByName(const char* name) const;

private:
void AddHeadersToRequest(const std::shared_ptr<Aws::Http::HttpRequest>& httpRequest, const Http::HeaderValueCollection& headerValues) const;
void AddContentBodyToRequest(const std::shared_ptr<Aws::Http::HttpRequest>& httpRequest,
const std::shared_ptr<Aws::IOStream>& body, bool needsContentMd5 = false) const;
void AddCommonHeaders(Aws::Http::HttpRequest& httpRequest) const;
void InitializeGlobalStatics();
void CleanupGlobalStatics();
std::shared_ptr<Aws::Http::HttpRequest> ConvertToRequestForPresigning(const Aws::AmazonWebServiceRequest& request, Aws::Http::URI& uri,
Aws::Http::HttpMethod method, const Aws::Http::QueryStringParameterCollection& extraParams) const;

std::shared_ptr<Aws::Http::HttpClient> m_httpClient;
Aws::Map<Aws::String, std::shared_ptr<Aws::Client::AWSAuthSigner>> m_signerMap;
std::shared_ptr<AWSErrorMarshaller> m_errorMarshaller;
std::shared_ptr<RetryStrategy> m_retryStrategy;
std::shared_ptr<Aws::Utils::RateLimits::RateLimiterInterface> m_writeRateLimiter;
std::shared_ptr<Aws::Utils::RateLimits::RateLimiterInterface> m_readRateLimiter;
Aws::String m_userAgent;
std::shared_ptr<Aws::Utils::Crypto::Hash> m_hash;
static std::atomic<int> s_refCount;
bool m_enableClockSkewAdjustment;
};

typedef Utils::Outcome<AmazonWebServiceResult<Utils::Json::JsonValue>, AWSError<CoreErrors>> JsonOutcome;


class AWS_CORE_API AWSJsonClient : public AWSClient
{
public:
typedef AWSClient BASECLASS;


AWSJsonClient(const Aws::Client::ClientConfiguration& configuration,
const std::shared_ptr<Aws::Client::AWSAuthSigner>& signer,
const std::shared_ptr<AWSErrorMarshaller>& errorMarshaller);


AWSJsonClient(const Aws::Client::ClientConfiguration& configuration,
const Aws::Map<Aws::String, std::shared_ptr<Aws::Client::AWSAuthSigner>>& signerMap,
const std::shared_ptr<AWSErrorMarshaller>& errorMarshaller);

virtual ~AWSJsonClient() = default;

protected:

virtual AWSError<CoreErrors> BuildAWSError(const std::shared_ptr<Aws::Http::HttpResponse>& response) const override;


JsonOutcome MakeRequest(const Aws::Http::URI& uri,
const Aws::AmazonWebServiceRequest& request,
Http::HttpMethod method = Http::HttpMethod::HTTP_POST,
const char* signerName = Aws::Auth::SIGV4_SIGNER) const;


JsonOutcome MakeRequest(const Aws::Http::URI& uri,
Http::HttpMethod method = Http::HttpMethod::HTTP_POST,
const char* signerName = Aws::Auth::SIGV4_SIGNER,
const char* requestName = nullptr) const;
};

typedef Utils::Outcome<AmazonWebServiceResult<Utils::Xml::XmlDocument>, AWSError<CoreErrors>> XmlOutcome;


class AWS_CORE_API AWSXMLClient : public AWSClient
{
public:

typedef AWSClient BASECLASS;

AWSXMLClient(const Aws::Client::ClientConfiguration& configuration,
const std::shared_ptr<Aws::Client::AWSAuthSigner>& signer,
const std::shared_ptr<AWSErrorMarshaller>& errorMarshaller);

AWSXMLClient(const Aws::Client::ClientConfiguration& configuration,
const Aws::Map<Aws::String, std::shared_ptr<Aws::Client::AWSAuthSigner>>& signerMap,
const std::shared_ptr<AWSErrorMarshaller>& errorMarshaller);

virtual ~AWSXMLClient() = default;

protected:

virtual AWSError<CoreErrors> BuildAWSError(const std::shared_ptr<Aws::Http::HttpResponse>& response) const override;


XmlOutcome MakeRequest(const Aws::Http::URI& uri,
const Aws::AmazonWebServiceRequest& request,
Http::HttpMethod method = Http::HttpMethod::HTTP_POST,
const char* signerName = Aws::Auth::SIGV4_SIGNER) const;



XmlOutcome MakeRequest(const Aws::Http::URI& uri,
Http::HttpMethod method = Http::HttpMethod::HTTP_POST,
const char* signerName = Aws::Auth::SIGV4_SIGNER,
const char* requesetName = nullptr) const;
};

} 
} 
