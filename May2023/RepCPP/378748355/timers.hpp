

#pragma once

#include "Manager/Manager.hpp"
#include <Impl/pool_impl.hpp>
#include <amx/amx.h>

struct PawnTimerHandler;

struct PawnTimerImpl : public Singleton<PawnTimerImpl>
{
friend struct PawnTimerHandler;

int setTimer(const char* callback, Milliseconds interval, bool repeating, AMX* amx);
int setTimerEx(const char* callback, Milliseconds interval, bool repeating, const char* fmt, AMX* amx, const cell* params);

ITimer* getTimer(int id) const
{
if (const auto& res = pool.find(id); res != pool.end())
{
return res->second;
}
return nullptr;
}

void killTimers(AMX* amx);

private:
Pair<size_t, PawnTimerHandler*> newTimer(const char* callback, Milliseconds interval, bool repeating, AMX* amx);
int newTimerExError(PawnTimerHandler* handler, AMX* amx, int err, StringView message);

size_t insert(ITimer* timer)
{
bool wrappedOnce = false;
while (pool.find(idx) != pool.end())
{
if (idx == UINT_MAX)
{
if (wrappedOnce)
{
return 0;
}
idx = 1;
wrappedOnce = true;
}
else
{
++idx;
}
}
pool.emplace(idx, timer);
size_t lastIdx = idx;
if (idx == UINT_MAX)
{
idx = 1;
}
else
{
++idx;
}
return lastIdx;
}

bool remove(uint32_t id)
{
return pool.erase(id);
}

FlatHashMap<uint32_t, ITimer*> pool;
uint32_t idx = 1;
};

struct PawnTimerHandler final : TimerTimeOutHandler, PoolIDProvider
{
AMX* amx;
HybridString<sNAMEMAX + 1> callback;
HybridString<64> fmt;
DynamicArray<cell> params;
DynamicArray<cell> data;

PawnTimerHandler(String callback, AMX* amx)
: amx(amx)
, callback(callback)
{
}

void timeout(ITimer& timer) override
{
if (!amx)
{
return;
}

const bool hasParams = !fmt.empty();

cell ret;
cell out;
cell* in;
int err = AMX_ERR_NONE;
if (hasParams)
{
if ((err = amx_Allot(amx, data.size(), &out, &in)) != AMX_ERR_NONE)
{
PawnManager::Get()->core->logLn(LogLevel::Error, "SetTimer(Ex): Not enough space in heap for %.*s timer: %s", PRINT_VIEW(callback), aux_StrError(err));
amx_RaiseError(amx, err);
return;
}
if (data.data())
{
memcpy(in, data.data(), data.size() * sizeof(cell));
}
for (size_t i = params.size(); i--;)
{
switch (fmt[i])
{
case 'a':
case 's':
case 'v':
amx_Push(amx, out + params[i]);
break;
default:
amx_Push(amx, params[i]);
break;
}
}
}

int funcidx;
if ((err = amx_FindPublic(amx, callback.data(), &funcidx)) == AMX_ERR_NONE && (err = amx_Exec(amx, &ret, funcidx)) == AMX_ERR_NONE)
{
if (hasParams)
{
for (size_t i = 0, len = params.size(); i != len; ++i)
{
switch (fmt[i])
{
case 'v':
data[params[i] / sizeof(cell)] = *((char*)in + params[i]);
break;
}
}
}
}
else
{
PawnManager::Get()->core->logLn(LogLevel::Error, "SetTimer(Ex): There was a problem in calling %.*s: %s", PRINT_VIEW(callback), aux_StrError(err));

}

if (hasParams)
{
amx_Release(amx, out);
}
}

void free(ITimer& timer) override
{
PawnTimerImpl::Get()->remove(poolID);
delete this;
}
};
