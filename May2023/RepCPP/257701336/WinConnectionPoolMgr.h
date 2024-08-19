

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/memory/stl/AWSMap.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/ResourceManager.h>

#include <utility>

namespace Aws
{
namespace Http
{


class AWS_CORE_API WinConnectionPoolMgr
{
public:

WinConnectionPoolMgr(void* iOpenHandle, unsigned maxConnectionsPerHost, long requestTimeout, long connectTimeout);


virtual ~WinConnectionPoolMgr();


void* AquireConnectionForHost(const Aws::String& host, uint16_t port);


void ReleaseConnectionForHost(const Aws::String& host, unsigned port, void* connection);


virtual const char* GetLogTag() const { return "ConnectionPoolMgr"; }

virtual void DoCloseHandle(void* handle) const = 0;
protected:

class AWS_CORE_API HostConnectionContainer
{
public:
uint16_t port;
Aws::Utils::ExclusiveOwnershipResourceManager<void*> hostConnections;
unsigned currentPoolSize;
};


void* GetOpenHandle() const { return m_iOpenHandle; }


long GetRequestTimeout() const { return m_requestTimeoutMs; }

long GetConnectTimeout() const { return m_connectTimeoutMs; }

void DoCleanup();

private:

virtual void* CreateNewConnection(const Aws::String& host, HostConnectionContainer& connectionContainer) const = 0;

WinConnectionPoolMgr(const WinConnectionPoolMgr&) = delete;
const WinConnectionPoolMgr& operator = (const WinConnectionPoolMgr&) = delete;
WinConnectionPoolMgr(const WinConnectionPoolMgr&&) = delete;
const WinConnectionPoolMgr& operator = (const WinConnectionPoolMgr&&) = delete;

bool CheckAndGrowPool(const Aws::String& host, HostConnectionContainer& connectionContainer);

void* m_iOpenHandle;
Aws::Map<Aws::String, HostConnectionContainer*> m_hostConnections;
std::mutex m_hostConnectionsMutex;
unsigned m_maxConnectionsPerHost;
long m_requestTimeoutMs;
long m_connectTimeoutMs;
std::mutex m_containerLock;
};

} 
} 
