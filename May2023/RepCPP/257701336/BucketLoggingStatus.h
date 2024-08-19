

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/LoggingEnabled.h>
#include <utility>

namespace Aws
{
namespace Utils
{
namespace Xml
{
class XmlNode;
} 
} 
namespace S3
{
namespace Model
{

class AWS_S3_API BucketLoggingStatus
{
public:
BucketLoggingStatus();
BucketLoggingStatus(const Aws::Utils::Xml::XmlNode& xmlNode);
BucketLoggingStatus& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const LoggingEnabled& GetLoggingEnabled() const{ return m_loggingEnabled; }


inline void SetLoggingEnabled(const LoggingEnabled& value) { m_loggingEnabledHasBeenSet = true; m_loggingEnabled = value; }


inline void SetLoggingEnabled(LoggingEnabled&& value) { m_loggingEnabledHasBeenSet = true; m_loggingEnabled = std::move(value); }


inline BucketLoggingStatus& WithLoggingEnabled(const LoggingEnabled& value) { SetLoggingEnabled(value); return *this;}


inline BucketLoggingStatus& WithLoggingEnabled(LoggingEnabled&& value) { SetLoggingEnabled(std::move(value)); return *this;}

private:

LoggingEnabled m_loggingEnabled;
bool m_loggingEnabledHasBeenSet;
};

} 
} 
} 
