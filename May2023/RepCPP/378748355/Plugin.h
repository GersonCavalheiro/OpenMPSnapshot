


#pragma once

#ifdef WIN32
#include <Windows.h>

#define PLUGIN_STD_CALL __stdcall
typedef HMODULE PLUGIN_MODULE;
#else
#include <dlfcn.h>
#include <stdlib.h>

#define PLUGIN_STD_CALL
typedef void* PLUGIN_MODULE;
#endif

#include "sdk.hpp"

#include <string>

#include <amx/amx.h>

class PawnPlugin
{
public:
PawnPlugin(std::string const& path, ICore* core);
~PawnPlugin();

void AmxLoad(AMX* amx);
void AmxUnload(AMX* amx);
void Unload();
void ProcessTick();

bool IsLoaded() const
{
return loaded_;
}

private:
typedef void(PLUGIN_STD_CALL* process_tick_f)();
typedef int(PLUGIN_STD_CALL* amx_load_f)(AMX*);
typedef int(PLUGIN_STD_CALL* amx_unload_f)(AMX*);
typedef unsigned int(PLUGIN_STD_CALL* supports_f)();
typedef int(PLUGIN_STD_CALL* load_f)(void const* const*);
typedef int(PLUGIN_STD_CALL* unload_f)();

ICore* serverCore;
bool loaded_ = false;

PLUGIN_MODULE pluginHandle_ = nullptr;

process_tick_f ProcessTick_ = nullptr;
amx_load_f AmxLoad_ = nullptr;
amx_unload_f AmxUnload_ = nullptr;
unload_f Unload_ = nullptr;
};

#undef PLUGIN_STD_CALL
