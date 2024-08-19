#pragma once

#include <iwasm/include/wasm_export.h>
#include <sgx.h>
#include <sgx_defs.h>

extern "C"
{
extern sgx_status_t SGX_CDECL ocallLogError(const char* msg);

#ifdef FAASM_SGX_DEBUG
void ocallLogDebug(const char* msg) { ; };
#else
extern sgx_status_t SGX_CDECL ocallLogDebug(const char* msg);
#endif

extern sgx_status_t SGX_CDECL ocallFaasmReadInput(int* returnValue,
uint8_t* buffer,
unsigned int bufferSize);

extern sgx_status_t SGX_CDECL
ocallFaasmWriteOutput(uint8_t* output, unsigned int outputSize);

extern sgx_status_t SGX_CDECL ocallFaasmChainName(unsigned int* returnValue,
const char* name,
const uint8_t* input,
unsigned int inputSize);

extern sgx_status_t SGX_CDECL ocallFaasmChainPtr(unsigned int* returnValue,
const int wasmFuncPtr,
const uint8_t* input,
unsigned int inputSize);

extern sgx_status_t SGX_CDECL ocallFaasmAwaitCall(unsigned int* returnValue,
unsigned int callId);

extern sgx_status_t SGX_CDECL
ocallFaasmAwaitCallOutput(unsigned int* returnValue,
unsigned int callId,
uint8_t* buffer,
unsigned int bufferSize);

extern sgx_status_t SGX_CDECL ocallSbrk(int32_t* returnValue,
int32_t increment);
}
