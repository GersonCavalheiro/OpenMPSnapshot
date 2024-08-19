

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <memory>

namespace Aws
{
namespace Auth
{

class AWS_CORE_API AWSCredentialsProviderChain : public AWSCredentialsProvider
{
public:
virtual ~AWSCredentialsProviderChain() = default;


virtual AWSCredentials GetAWSCredentials();

protected:

AWSCredentialsProviderChain() = default;


void AddProvider(const std::shared_ptr<AWSCredentialsProvider>& provider) { m_providerChain.push_back(provider); }

private:            
Aws::Vector<std::shared_ptr<AWSCredentialsProvider> > m_providerChain;
};


class AWS_CORE_API DefaultAWSCredentialsProviderChain : public AWSCredentialsProviderChain
{
public:

DefaultAWSCredentialsProviderChain();
};

} 
} 
