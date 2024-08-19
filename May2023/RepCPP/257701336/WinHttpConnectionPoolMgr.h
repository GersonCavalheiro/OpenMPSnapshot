

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/http/windows/WinConnectionPoolMgr.h>

namespace Aws
{
namespace Http
{


class AWS_CORE_API WinHttpConnectionPoolMgr : public WinConnectionPoolMgr
{
public:

WinHttpConnectionPoolMgr(void* iOpenHandle, unsigned maxConnectionsPerHost, long requestTimeout, long connectTimeout);

virtual ~WinHttpConnectionPoolMgr();


const char* GetLogTag() const { return "WinHttpConnectionPoolMgr"; }

private:
virtual void DoCloseHandle(void* handle) const override;
virtual void* CreateNewConnection(const Aws::String& host, HostConnectionContainer& connectionContainer) const override;

WinHttpConnectionPoolMgr(const WinHttpConnectionPoolMgr&) = delete;
const WinHttpConnectionPoolMgr& operator = (const WinHttpConnectionPoolMgr&) = delete;
WinHttpConnectionPoolMgr(const WinHttpConnectionPoolMgr&&) = delete;
const WinHttpConnectionPoolMgr& operator = (const WinHttpConnectionPoolMgr&&) = delete;

};

} 
} 
