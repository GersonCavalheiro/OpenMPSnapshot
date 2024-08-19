

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/memory/AWSMemory.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <memory>

namespace Aws
{
namespace Http
{
class HttpClient;
} 

namespace Internal
{

class AWS_CORE_API AWSHttpResourceClient
{
public:

AWSHttpResourceClient(const char* logtag = "AWSHttpResourceClient");

AWSHttpResourceClient& operator =(const AWSHttpResourceClient& rhs) = delete;
AWSHttpResourceClient(const AWSHttpResourceClient& rhs) = delete;
AWSHttpResourceClient& operator =(const AWSHttpResourceClient&& rhs) = delete;
AWSHttpResourceClient(const AWSHttpResourceClient&& rhs) = delete;

virtual ~AWSHttpResourceClient();


virtual Aws::String GetResource(const char* endpoint, const char* resourcePath) const;

protected:
Aws::String m_logtag;

private:         
std::shared_ptr<Http::HttpClient> m_httpClient;
};


class AWS_CORE_API EC2MetadataClient : public AWSHttpResourceClient
{
public:

EC2MetadataClient(const char* endpoint = "http:

EC2MetadataClient& operator =(const EC2MetadataClient& rhs) = delete;
EC2MetadataClient(const EC2MetadataClient& rhs) = delete;
EC2MetadataClient& operator =(const EC2MetadataClient&& rhs) = delete;
EC2MetadataClient(const EC2MetadataClient&& rhs) = delete;

virtual ~EC2MetadataClient();

using AWSHttpResourceClient::GetResource;


virtual Aws::String GetResource(const char* resourcePath) const;


virtual Aws::String GetDefaultCredentials() const;


virtual Aws::String GetCurrentRegion() const;

private:
Aws::String m_endpoint;
};


class AWS_CORE_API ECSCredentialsClient : public AWSHttpResourceClient
{
public:

ECSCredentialsClient(const char* resourcePath, const char* endpoint = "http:
ECSCredentialsClient& operator =(ECSCredentialsClient& rhs) = delete;
ECSCredentialsClient(const ECSCredentialsClient& rhs) = delete;
ECSCredentialsClient& operator =(ECSCredentialsClient&& rhs) = delete;
ECSCredentialsClient(const ECSCredentialsClient&& rhs) = delete;

virtual ~ECSCredentialsClient();


virtual Aws::String GetECSCredentials() const 
{
return this->GetResource(m_endpoint.c_str(), m_resourcePath.c_str());
}

private:
Aws::String m_resourcePath;
Aws::String m_endpoint;
};

} 
} 
