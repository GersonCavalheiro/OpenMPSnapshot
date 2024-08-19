

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/http/windows/WinConnectionPoolMgr.h>

namespace Aws
{
namespace Http
{


class AWS_CORE_API WinINetConnectionPoolMgr : public WinConnectionPoolMgr
{
public:

WinINetConnectionPoolMgr(void* iOpenHandle, unsigned maxConnectionsPerHost, long requestTimeout, long connectTimeout);

virtual ~WinINetConnectionPoolMgr();


const char* GetLogTag() const override { return "WinInetConnectionPoolMgr"; }

private:
virtual void DoCloseHandle(void* handle) const override;

virtual void* CreateNewConnection(const Aws::String& host, HostConnectionContainer& connectionContainer) const override;

WinINetConnectionPoolMgr(const WinINetConnectionPoolMgr&) = delete;
const WinINetConnectionPoolMgr& operator = (const WinINetConnectionPoolMgr&) = delete;
WinINetConnectionPoolMgr(const WinINetConnectionPoolMgr&&) = delete;
const WinINetConnectionPoolMgr& operator = (const WinINetConnectionPoolMgr&&) = delete;

};

} 
} 
