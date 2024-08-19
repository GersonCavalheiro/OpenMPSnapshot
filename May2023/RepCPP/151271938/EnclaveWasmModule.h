#pragma once

#include <wamr/WAMRModuleMixin.h>

#include <iwasm/aot/aot_runtime.h>
#include <wasm_runtime_common.h>

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#define ONE_KB_BYTES 1024
#define ONE_MB_BYTES (1024 * 1024)

#define FAASM_SGX_WAMR_HEAP_SIZE (32 * ONE_MB_BYTES)
#define FAASM_SGX_WAMR_MODULE_ERROR_BUFFER_SIZE 128
#define FAASM_SGX_WAMR_INSTANCE_DEFAULT_HEAP_SIZE (8 * ONE_KB_BYTES)
#define FAASM_SGX_WAMR_INSTANCE_DEFAULT_STACK_SIZE (8 * ONE_KB_BYTES)

#define WASM_CTORS_FUNC_NAME "__wasm_call_ctors"
#define WASM_ENTRY_FUNC "_start"

namespace wasm {


class EnclaveWasmModule : public WAMRModuleMixin<EnclaveWasmModule>
{
public:
static bool initialiseWAMRGlobally();

EnclaveWasmModule();

~EnclaveWasmModule();

bool loadWasm(void* wasmOpCodePtr, uint32_t wasmOpCodeSize);

bool callFunction(uint32_t argcIn, char** argvIn);

WASMModuleInstanceCommon* getModuleInstance();


uint32_t getArgc();

std::vector<std::string> getArgv();

size_t getArgvBufferSize();

private:
char errorBuffer[FAASM_SGX_WAMR_MODULE_ERROR_BUFFER_SIZE];

WASMModuleCommon* wasmModule;
WASMModuleInstanceCommon* moduleInstance;

uint32_t argc;
std::vector<std::string> argv;
size_t argvBufferSize;

void prepareArgcArgv(uint32_t argcIn, char** argvIn);
};

extern std::unordered_map<uint32_t, std::shared_ptr<wasm::EnclaveWasmModule>>
moduleMap;
extern std::mutex moduleMapMutex;

std::shared_ptr<wasm::EnclaveWasmModule> getExecutingEnclaveWasmModule(
wasm_exec_env_t execEnv);
}

#define GET_EXECUTING_MODULE_AND_CHECK(execEnv)                                \
std::shared_ptr<wasm::EnclaveWasmModule> module =                          \
wasm::getExecutingEnclaveWasmModule(execEnv);                            \
if (module == nullptr) {                                                   \
ocallLogError(                                                         \
"Error linking execution environment to registered modules");        \
return 1;                                                              \
}
