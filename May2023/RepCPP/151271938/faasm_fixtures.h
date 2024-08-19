#pragma once

#include "fixtures.h"

#include <codegen/MachineCodeGenerator.h>
#include <conf/FaasmConfig.h>
#include <storage/FileLoader.h>
#include <storage/S3Wrapper.h>
#include <storage/SharedFiles.h>
#include <wavm/WAVMWasmModule.h>

#include <faabric/util/files.h>

namespace tests {


class FaasmConfTestFixture
{
public:
FaasmConfTestFixture()
: conf(conf::getFaasmConfig())
{}
~FaasmConfTestFixture() { conf.reset(); }

protected:
conf::FaasmConfig& conf;
};


class S3TestFixture
{
public:
S3TestFixture()
: conf(conf::getFaasmConfig())
{
conf.s3Bucket = "faasm-test";
s3.createBucket(conf.s3Bucket);
};

~S3TestFixture()
{
s3.deleteBucket(conf.s3Bucket);
conf.reset();
};

protected:
storage::S3Wrapper s3;
conf::FaasmConfig& conf;
};


class SharedFilesTestFixture : public S3TestFixture
{
public:
SharedFilesTestFixture()
: loader(storage::getFileLoader())
{
storage::SharedFiles::clear();
}

~SharedFilesTestFixture() { storage::SharedFiles::clear(); }

protected:
storage::FileLoader& loader;
};


class IRModuleCacheTestFixture
{
public:
IRModuleCacheTestFixture();
~IRModuleCacheTestFixture();
};


class WAVMModuleCacheTestFixture
{
public:
WAVMModuleCacheTestFixture();
~WAVMModuleCacheTestFixture();

protected:
wasm::WAVMModuleCache& moduleCache;
};


class FunctionExecTestFixture
: public SchedulerTestFixture
, public WAVMModuleCacheTestFixture
, public IRModuleCacheTestFixture
, public ExecutorContextTestFixture
{
public:
FunctionExecTestFixture() {}
~FunctionExecTestFixture() {}
};


class MultiRuntimeFunctionExecTestFixture
: public FaasmConfTestFixture
, public FunctionExecTestFixture
{
public:
MultiRuntimeFunctionExecTestFixture() {}
~MultiRuntimeFunctionExecTestFixture() {}
};


class FunctionLoaderTestFixture : public S3TestFixture
{
public:
FunctionLoaderTestFixture()
: loader(storage::getFileLoader())
, gen(codegen::getMachineCodeGenerator())
{
msgA = faabric::util::messageFactory("demo", "hello");
msgB = faabric::util::messageFactory("demo", "echo");
wasmBytesA = loader.loadFunctionWasm(msgA);
wasmBytesB = loader.loadFunctionWasm(msgB);
msgA.set_inputdata(wasmBytesA.data(), wasmBytesA.size());
msgB.set_inputdata(wasmBytesB.data(), wasmBytesB.size());

objBytesA = loader.loadFunctionObjectFile(msgA);
objBytesB = loader.loadFunctionObjectFile(msgB);

hashBytesA = loader.loadFunctionObjectHash(msgA);
hashBytesB = loader.loadFunctionObjectHash(msgB);

localSharedObjFile =
conf.runtimeFilesDir + "/lib/python3.8/lib-dynload/syslog.so";
sharedObjWasm = faabric::util::readFileToBytes(localSharedObjFile);

conf.functionDir = "/tmp/func";
conf.objectFileDir = "/tmp/obj";
conf.sharedFilesDir = "/tmp/shared";
}

void uploadTestWasm()
{
loader.uploadFunction(msgA);
gen.codegenForFunction(msgA);
loader.uploadFunction(msgB);
gen.codegenForFunction(msgB);
}

~FunctionLoaderTestFixture() { loader.clearLocalCache(); }

protected:
storage::FileLoader& loader;
codegen::MachineCodeGenerator& gen;

faabric::Message msgA;
faabric::Message msgB;

std::vector<uint8_t> wasmBytesA;
std::vector<uint8_t> wasmBytesB;
std::vector<uint8_t> objBytesA;
std::vector<uint8_t> objBytesB;
std::vector<uint8_t> hashBytesA;
std::vector<uint8_t> hashBytesB;

std::string localSharedObjFile;
std::vector<uint8_t> sharedObjWasm;
};
}
