

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/logging/LogLevel.h>
#include <aws/core/utils/logging/AWSLogging.h>
#include <aws/core/utils/logging/LogSystemInterface.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>


#ifdef DISABLE_AWS_LOGGING

#define AWS_LOG(level, tag, ...) 
#define AWS_LOG_FATAL(tag, ...) 
#define AWS_LOG_ERROR(tag, ...) 
#define AWS_LOG_WARN(tag, ...) 
#define AWS_LOG_INFO(tag, ...) 
#define AWS_LOG_DEBUG(tag, ...) 
#define AWS_LOG_TRACE(tag, ...) 

#define AWS_LOGSTREAM(level, tag, streamExpression) 
#define AWS_LOGSTREAM_FATAL(tag, streamExpression)
#define AWS_LOGSTREAM_ERROR(tag, streamExpression)
#define AWS_LOGSTREAM_WARN(tag, streamExpression)
#define AWS_LOGSTREAM_INFO(tag, streamExpression)
#define AWS_LOGSTREAM_DEBUG(tag, streamExpression)
#define AWS_LOGSTREAM_TRACE(tag, streamExpression)

#else

#define AWS_LOG(level, tag, ...) \
{ \
Aws::Utils::Logging::LogSystemInterface* logSystem = Aws::Utils::Logging::GetLogSystem(); \
if ( logSystem && logSystem->GetLogLevel() >= level ) \
{ \
logSystem->Log(level, tag, __VA_ARGS__); \
} \
}

#define AWS_LOG_FATAL(tag, ...) \
{ \
Aws::Utils::Logging::LogSystemInterface* logSystem = Aws::Utils::Logging::GetLogSystem(); \
if ( logSystem && logSystem->GetLogLevel() >= Aws::Utils::Logging::LogLevel::Fatal ) \
{ \
logSystem->Log(Aws::Utils::Logging::LogLevel::Fatal, tag, __VA_ARGS__); \
} \
}

#define AWS_LOG_ERROR(tag, ...) \
{ \
Aws::Utils::Logging::LogSystemInterface* logSystem = Aws::Utils::Logging::GetLogSystem(); \
if ( logSystem && logSystem->GetLogLevel() >= Aws::Utils::Logging::LogLevel::Error ) \
{ \
logSystem->Log(Aws::Utils::Logging::LogLevel::Error, tag, __VA_ARGS__); \
} \
}

#define AWS_LOG_WARN(tag, ...) \
{ \
Aws::Utils::Logging::LogSystemInterface* logSystem = Aws::Utils::Logging::GetLogSystem(); \
if ( logSystem && logSystem->GetLogLevel() >= Aws::Utils::Logging::LogLevel::Warn ) \
{ \
logSystem->Log(Aws::Utils::Logging::LogLevel::Warn, tag, __VA_ARGS__); \
} \
}

#define AWS_LOG_INFO(tag, ...) \
{ \
Aws::Utils::Logging::LogSystemInterface* logSystem = Aws::Utils::Logging::GetLogSystem(); \
if ( logSystem && logSystem->GetLogLevel() >= Aws::Utils::Logging::LogLevel::Info ) \
{ \
logSystem->Log(Aws::Utils::Logging::LogLevel::Info, tag, __VA_ARGS__); \
} \
}

#define AWS_LOG_DEBUG(tag, ...) \
{ \
Aws::Utils::Logging::LogSystemInterface* logSystem = Aws::Utils::Logging::GetLogSystem(); \
if ( logSystem && logSystem->GetLogLevel() >= Aws::Utils::Logging::LogLevel::Debug ) \
{ \
logSystem->Log(Aws::Utils::Logging::LogLevel::Debug, tag, __VA_ARGS__); \
} \
}

#define AWS_LOG_TRACE(tag, ...) \
{ \
Aws::Utils::Logging::LogSystemInterface* logSystem = Aws::Utils::Logging::GetLogSystem(); \
if ( logSystem && logSystem->GetLogLevel() >= Aws::Utils::Logging::LogLevel::Trace ) \
{ \
logSystem->Log(Aws::Utils::Logging::LogLevel::Trace, tag, __VA_ARGS__); \
} \
}

#define AWS_LOGSTREAM(level, tag, streamExpression) \
{ \
Aws::Utils::Logging::LogSystemInterface* logSystem = Aws::Utils::Logging::GetLogSystem(); \
if ( logSystem && logSystem->GetLogLevel() >= level ) \
{ \
Aws::OStringStream logStream; \
logStream << streamExpression; \
logSystem->LogStream( logLevel, tag, logStream ); \
} \
}

#define AWS_LOGSTREAM_FATAL(tag, streamExpression) \
{ \
Aws::Utils::Logging::LogSystemInterface* logSystem = Aws::Utils::Logging::GetLogSystem(); \
if ( logSystem && logSystem->GetLogLevel() >= Aws::Utils::Logging::LogLevel::Fatal ) \
{ \
Aws::OStringStream logStream; \
logStream << streamExpression; \
logSystem->LogStream( Aws::Utils::Logging::LogLevel::Fatal, tag, logStream ); \
} \
}

#define AWS_LOGSTREAM_ERROR(tag, streamExpression) \
{ \
Aws::Utils::Logging::LogSystemInterface* logSystem = Aws::Utils::Logging::GetLogSystem(); \
if ( logSystem && logSystem->GetLogLevel() >= Aws::Utils::Logging::LogLevel::Error ) \
{ \
Aws::OStringStream logStream; \
logStream << streamExpression; \
logSystem->LogStream( Aws::Utils::Logging::LogLevel::Error, tag, logStream ); \
} \
}

#define AWS_LOGSTREAM_WARN(tag, streamExpression) \
{ \
Aws::Utils::Logging::LogSystemInterface* logSystem = Aws::Utils::Logging::GetLogSystem(); \
if ( logSystem && logSystem->GetLogLevel() >= Aws::Utils::Logging::LogLevel::Warn ) \
{ \
Aws::OStringStream logStream; \
logStream << streamExpression; \
logSystem->LogStream( Aws::Utils::Logging::LogLevel::Warn, tag, logStream ); \
} \
}

#define AWS_LOGSTREAM_INFO(tag, streamExpression) \
{ \
Aws::Utils::Logging::LogSystemInterface* logSystem = Aws::Utils::Logging::GetLogSystem(); \
if ( logSystem && logSystem->GetLogLevel() >= Aws::Utils::Logging::LogLevel::Info ) \
{ \
Aws::OStringStream logStream; \
logStream << streamExpression; \
logSystem->LogStream( Aws::Utils::Logging::LogLevel::Info, tag, logStream ); \
} \
}

#define AWS_LOGSTREAM_DEBUG(tag, streamExpression) \
{ \
Aws::Utils::Logging::LogSystemInterface* logSystem = Aws::Utils::Logging::GetLogSystem(); \
if ( logSystem && logSystem->GetLogLevel() >= Aws::Utils::Logging::LogLevel::Debug ) \
{ \
Aws::OStringStream logStream; \
logStream << streamExpression; \
logSystem->LogStream( Aws::Utils::Logging::LogLevel::Debug, tag, logStream ); \
} \
}

#define AWS_LOGSTREAM_TRACE(tag, streamExpression) \
{ \
Aws::Utils::Logging::LogSystemInterface* logSystem = Aws::Utils::Logging::GetLogSystem(); \
if ( logSystem && logSystem->GetLogLevel() >= Aws::Utils::Logging::LogLevel::Trace ) \
{ \
Aws::OStringStream logStream; \
logStream << streamExpression; \
logSystem->LogStream( Aws::Utils::Logging::LogLevel::Trace, tag, logStream ); \
} \
}

#endif 
