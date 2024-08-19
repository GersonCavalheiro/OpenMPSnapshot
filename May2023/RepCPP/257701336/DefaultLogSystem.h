

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/logging/FormattedLogSystem.h>
#include <aws/core/utils/logging/LogLevel.h>
#include <aws/core/utils/memory/stl/AWSQueue.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSStreamFwd.h>

#include <thread>
#include <memory>
#include <mutex>
#include <atomic>
#include <condition_variable>

namespace Aws
{
namespace Utils
{
namespace Logging
{

class AWS_CORE_API DefaultLogSystem : public FormattedLogSystem
{
public:
using Base = FormattedLogSystem;


DefaultLogSystem(LogLevel logLevel, const std::shared_ptr<Aws::OStream>& logFile);

DefaultLogSystem(LogLevel logLevel, const Aws::String& filenamePrefix);
virtual ~DefaultLogSystem();


struct LogSynchronizationData
{
public:
LogSynchronizationData() : m_stopLogging(false) {}

std::mutex m_logQueueMutex;
std::condition_variable m_queueSignal;
Aws::Queue<Aws::String> m_queuedLogMessages;
std::atomic<bool> m_stopLogging;

private:
LogSynchronizationData(const LogSynchronizationData& rhs) = delete;
LogSynchronizationData& operator =(const LogSynchronizationData& rhs) = delete;
};

protected:

virtual void ProcessFormattedStatement(Aws::String&& statement) override;

private:
DefaultLogSystem(const DefaultLogSystem& rhs) = delete;
DefaultLogSystem& operator =(const DefaultLogSystem& rhs) = delete;

LogSynchronizationData m_syncData;

std::thread m_loggingThread;
};

} 
} 
} 
