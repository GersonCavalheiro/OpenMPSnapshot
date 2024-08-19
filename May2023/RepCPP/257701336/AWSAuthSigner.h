

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/Region.h>
#include <aws/core/utils/memory/AWSMemory.h>
#include <aws/core/utils/memory/stl/AWSSet.h>
#include <aws/core/utils/DateTime.h>
#include <aws/core/utils/Array.h>

#include <memory>
#include <atomic>
#include <mutex>
#include <chrono>

namespace Aws
{
namespace Http
{
class HttpClientFactory;
class HttpRequest;
} 

namespace Utils
{
namespace Crypto
{
class Sha256;
class Sha256HMAC;
} 
} 

namespace Auth
{
class AWSCredentials;
class AWSCredentialsProvider;
AWS_CORE_API extern const char SIGV4_SIGNER[];
AWS_CORE_API extern const char NULL_SIGNER[];
} 

namespace Client
{
struct ClientConfiguration;


class AWS_CORE_API AWSAuthSigner
{
public:
AWSAuthSigner() : m_clockSkew() { m_clockSkew.store(std::chrono::milliseconds(0L)); }
virtual ~AWSAuthSigner() = default;


virtual bool SignRequest(Aws::Http::HttpRequest& request) const = 0;


virtual bool SignRequest(Aws::Http::HttpRequest& request, bool signBody) const { AWS_UNREFERENCED_PARAM(signBody); return SignRequest(request); }


virtual bool PresignRequest(Aws::Http::HttpRequest& request, long long expirationInSeconds) const = 0;


virtual bool PresignRequest(Aws::Http::HttpRequest& request, const char* region, long long expirationInSeconds = 0) const = 0;


virtual bool PresignRequest(Aws::Http::HttpRequest& request, const char* region, const char* serviceName, long long expirationInSeconds = 0) const = 0;


virtual const char* GetName() const = 0;


virtual void SetClockSkew(const std::chrono::milliseconds& clockSkew) { m_clockSkew = clockSkew; }


virtual Aws::Utils::DateTime GetSigningTimestamp() const { return Aws::Utils::DateTime::Now() + GetClockSkewOffset(); }

protected:            
virtual std::chrono::milliseconds GetClockSkewOffset() const { return m_clockSkew.load(); }

std::atomic<std::chrono::milliseconds> m_clockSkew;
};


class AWS_CORE_API AWSAuthV4Signer : public AWSAuthSigner
{

public:

enum class PayloadSigningPolicy
{

RequestDependent,

Always,

Never
};

AWSAuthV4Signer(const std::shared_ptr<Auth::AWSCredentialsProvider>& credentialsProvider,
const char* serviceName, const Aws::String& region, PayloadSigningPolicy signingPolicy = PayloadSigningPolicy::RequestDependent,
bool urlEscapePath = true);

virtual ~AWSAuthV4Signer();


const char* GetName() const override { return Aws::Auth::SIGV4_SIGNER; }


bool SignRequest(Aws::Http::HttpRequest& request) const override;


bool SignRequest(Aws::Http::HttpRequest& request, bool signBody) const override;


bool PresignRequest(Aws::Http::HttpRequest& request, long long expirationInSeconds = 0) const override;


bool PresignRequest(Aws::Http::HttpRequest& request, const char* region, long long expirationInSeconds = 0) const override;


bool PresignRequest(Aws::Http::HttpRequest& request, const char* region, const char* serviceName, long long expirationInSeconds = 0) const override;

protected:
bool m_includeSha256HashHeader;

private:
Aws::String GenerateSignature(const Aws::Auth::AWSCredentials& credentials, const Aws::String& stringToSign, const Aws::String& simpleDate) const;
Aws::String ComputePayloadHash(Aws::Http::HttpRequest&) const;
Aws::String GenerateStringToSign(const Aws::String& dateValue, const Aws::String& simpleDate, const Aws::String& canonicalRequestHash) const;
const Aws::Utils::ByteBuffer& ComputeLongLivedHash(const Aws::String& secretKey, const Aws::String& simpleDate) const;

bool ShouldSignHeader(const Aws::String& header) const;

std::shared_ptr<Auth::AWSCredentialsProvider> m_credentialsProvider;
Aws::String m_serviceName;
Aws::String m_region;
Aws::UniquePtr<Aws::Utils::Crypto::Sha256> m_hash;
Aws::UniquePtr<Aws::Utils::Crypto::Sha256HMAC> m_HMAC;

Aws::Set<Aws::String> m_unsignedHeaders;

mutable Aws::Utils::ByteBuffer m_partialSignature;
mutable Aws::String m_currentDateStr;
mutable Aws::String m_currentSecretKey;
mutable std::mutex m_partialSignatureLock;
PayloadSigningPolicy m_payloadSigningPolicy;
bool m_urlEscapePath;
};



class AWS_CORE_API AWSNullSigner : public AWSAuthSigner
{
public:

const char* GetName() const override { return Aws::Auth::NULL_SIGNER; }


bool SignRequest(Aws::Http::HttpRequest&) const override { return true; }


bool SignRequest(Aws::Http::HttpRequest&, bool) const override { return true; }


bool PresignRequest(Aws::Http::HttpRequest&, long long) const override { return false; }


bool PresignRequest(Aws::Http::HttpRequest&, const char*, long long) const override { return false; }


bool PresignRequest(Aws::Http::HttpRequest&, const char*, const char*, long long) const override { return false; }
};

} 
} 

