#pragma once

#include <wasm_export.h>

#include <stdexcept>
#include <string>
#include <vector>


template<typename T>
struct WAMRModuleMixin
{
T& underlying() { return static_cast<T&>(*this); }


void validateNativePointer(void* nativePtr, int size)
{
auto moduleInstance = this->underlying().getModuleInstance();
bool success =
wasm_runtime_validate_native_addr(moduleInstance, nativePtr, size);

if (!success) {
throw std::runtime_error("Failed validating native pointer!");
}
}

void* wasmOffsetToNativePointer(uint32_t wasmOffset)
{
auto moduleInstance = this->underlying().getModuleInstance();
return wasm_runtime_addr_app_to_native(moduleInstance, wasmOffset);
}

uint32_t nativePointerToWasmOffset(void* nativePtr)
{
auto moduleInstance = this->underlying().getModuleInstance();
return wasm_runtime_addr_native_to_app(moduleInstance, nativePtr);
}

uint32_t wasmModuleMalloc(size_t size, void** nativePtr)
{
auto moduleInstance = this->underlying().getModuleInstance();
uint32_t wasmOffset =
wasm_runtime_module_malloc(moduleInstance, size, nativePtr);

if (wasmOffset == 0 || nativePtr == nullptr) {
throw std::runtime_error(
"Failed malloc-ing memory in WASM module!");
}

return wasmOffset;
}

void writeStringArrayToMemory(const std::vector<std::string>& strings,
uint32_t* strOffsets,
char* strBuffer)
{
validateNativePointer(strOffsets, strings.size() * sizeof(uint32_t));

char* nextBuffer = strBuffer;
for (size_t i = 0; i < strings.size(); i++) {
const std::string& thisStr = strings.at(i);

validateNativePointer(nextBuffer, thisStr.size() + 1);

std::copy(thisStr.begin(), thisStr.end(), nextBuffer);
nextBuffer[thisStr.size()] = '\0';
strOffsets[i] = nativePointerToWasmOffset(nextBuffer);

nextBuffer += thisStr.size() + 1;
}
}


void writeArgvToWamrMemory(uint32_t* argvOffsetsWasm, char* argvBuffWasm)
{
writeStringArrayToMemory(
this->underlying().getArgv(), argvOffsetsWasm, argvBuffWasm);
}
};
