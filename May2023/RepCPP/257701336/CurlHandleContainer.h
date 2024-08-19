

#pragma once

#include <aws/core/utils/ResourceManager.h>

#include <utility>
#include <curl/curl.h>

namespace Aws
{
namespace Http
{


class CurlHandleContainer
{
public:

CurlHandleContainer(unsigned maxSize = 50, long requestTimeout = 3000, long connectTimeout = 1000);
~CurlHandleContainer();


CURL* AcquireCurlHandle();

void ReleaseCurlHandle(CURL* handle);

private:
CurlHandleContainer(const CurlHandleContainer&) = delete;
const CurlHandleContainer& operator = (const CurlHandleContainer&) = delete;
CurlHandleContainer(const CurlHandleContainer&&) = delete;
const CurlHandleContainer& operator = (const CurlHandleContainer&&) = delete;

bool CheckAndGrowPool();
void SetDefaultOptionsOnHandle(CURL* handle);

Aws::Utils::ExclusiveOwnershipResourceManager<CURL*> m_handleContainer;
unsigned m_maxPoolSize;
unsigned long m_requestTimeout;
unsigned long m_connectTimeout;
unsigned m_poolSize;
std::mutex m_containerLock;
};

} 
} 

