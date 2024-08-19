
#pragma once

#include <aws/core/utils/logging/LogLevel.h>
#include <aws/core/utils/logging/LogSystemInterface.h>
#include <aws/core/utils/memory/MemorySystemInterface.h>
#include <aws/core/utils/crypto/Factories.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/Core_EXPORTS.h>

namespace Aws
{
static const char* DEFAULT_LOG_PREFIX = "aws_sdk_";


struct LoggingOptions
{
LoggingOptions() : logLevel(Aws::Utils::Logging::LogLevel::Off), defaultLogPrefix(DEFAULT_LOG_PREFIX)
{ }


Aws::Utils::Logging::LogLevel logLevel;


const char* defaultLogPrefix;


std::function<std::shared_ptr<Aws::Utils::Logging::LogSystemInterface>()> logger_create_fn;
};


struct MemoryManagementOptions
{
MemoryManagementOptions() : memoryManager(nullptr)
{ }


Aws::Utils::Memory::MemorySystemInterface* memoryManager;
};


struct HttpOptions
{
HttpOptions() : initAndCleanupCurl(true), installSigPipeHandler(false)
{ }


std::function<std::shared_ptr<Aws::Http::HttpClientFactory>()> httpClientFactory_create_fn;

bool initAndCleanupCurl;

bool installSigPipeHandler;
};


struct CryptoOptions
{
CryptoOptions() : initAndCleanupOpenSSL(true)
{ }


std::function<std::shared_ptr<Aws::Utils::Crypto::HashFactory>()> md5Factory_create_fn;

std::function<std::shared_ptr<Aws::Utils::Crypto::HashFactory>()> sha256Factory_create_fn;

std::function<std::shared_ptr<Aws::Utils::Crypto::HMACFactory>()> sha256HMACFactory_create_fn;

std::function<std::shared_ptr<Aws::Utils::Crypto::SymmetricCipherFactory>()> aes_CBCFactory_create_fn;

std::function<std::shared_ptr<Aws::Utils::Crypto::SymmetricCipherFactory>()> aes_CTRFactory_create_fn;

std::function<std::shared_ptr<Aws::Utils::Crypto::SymmetricCipherFactory>()> aes_GCMFactory_create_fn;

std::function<std::shared_ptr<Aws::Utils::Crypto::SymmetricCipherFactory>()> aes_KeyWrapFactory_create_fn;

std::function<std::shared_ptr<Aws::Utils::Crypto::SecureRandomFactory>()> secureRandomFactory_create_fn;

bool initAndCleanupOpenSSL;
};


struct SDKOptions
{

LoggingOptions loggingOptions;

MemoryManagementOptions memoryManagementOptions;

HttpOptions httpOptions;

CryptoOptions cryptoOptions;
};


AWS_CORE_API void InitAPI(const SDKOptions& options);


AWS_CORE_API void ShutdownAPI(const SDKOptions& options);
}

