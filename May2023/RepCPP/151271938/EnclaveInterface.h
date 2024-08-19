#pragma once

#include <enclave/error.h>

#include <storage/FileLoader.h>
#include <storage/FileSystem.h>
#include <wasm/WasmExecutionContext.h>
#include <wasm/WasmModule.h>

#include <sgx.h>
#include <sgx_urts.h>

namespace wasm {

class EnclaveInterface final : public WasmModule
{
public:
explicit EnclaveInterface();

~EnclaveInterface() override;

void doBindToFunction(faabric::Message& msg, bool cache) override;

bool unbindFunction();

int32_t executeFunction(faabric::Message& msg) override;

size_t getMemorySizeBytes() override;

size_t getMaxMemoryPages() override;

uint8_t* getMemoryBase() override;

private:
uint32_t interfaceId = 0;
};
}
