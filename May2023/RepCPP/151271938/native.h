#pragma once

#include <cstdint>
#include <lib_export.h>

#define REG_NATIVE_FUNC(func_name, signature)                                  \
{                                                                          \
#func_name, (void*)func_name##_wrapper, signature, nullptr             \
}

#define REG_WASI_NATIVE_FUNC(func_name, signature)                             \
{                                                                          \
#func_name, (void*)wasi_##func_name, signature, nullptr                \
}



namespace wasm {
void initialiseWAMRNatives();

uint32_t getFaasmDynlinkApi(NativeSymbol** nativeSymbols);

uint32_t getFaasmEnvApi(NativeSymbol** nativeSymbols);

uint32_t getFaasmFilesystemApi(NativeSymbol** nativeSymbols);

uint32_t getFaasmFunctionsApi(NativeSymbol** nativeSymbols);

uint32_t getFaasmMemoryApi(NativeSymbol** nativeSymbols);

uint32_t getFaasmMpiApi(NativeSymbol** nativeSymbols);

uint32_t getFaasmProcessApi(NativeSymbol** nativeSymbols);

uint32_t getFaasmPthreadApi(NativeSymbol** nativeSymbols);

uint32_t getFaasmSignalApi(NativeSymbol** nativeSymbols);

uint32_t getFaasmStateApi(NativeSymbol** nativeSymbols);

uint32_t getFaasmStubs(NativeSymbol** nativeSymbols);


uint32_t getFaasmWasiEnvApi(NativeSymbol** nativeSymbols);

uint32_t getFaasmWasiFilesystemApi(NativeSymbol** nativeSymbols);

uint32_t getFaasmWasiTimingApi(NativeSymbol** nativeSymbols);
}
