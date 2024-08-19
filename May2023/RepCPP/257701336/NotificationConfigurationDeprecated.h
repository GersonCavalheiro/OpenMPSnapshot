

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/TopicConfigurationDeprecated.h>
#include <aws/s3/model/QueueConfigurationDeprecated.h>
#include <aws/s3/model/CloudFunctionConfiguration.h>
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

class AWS_S3_API NotificationConfigurationDeprecated
{
public:
NotificationConfigurationDeprecated();
NotificationConfigurationDeprecated(const Aws::Utils::Xml::XmlNode& xmlNode);
NotificationConfigurationDeprecated& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const TopicConfigurationDeprecated& GetTopicConfiguration() const{ return m_topicConfiguration; }


inline void SetTopicConfiguration(const TopicConfigurationDeprecated& value) { m_topicConfigurationHasBeenSet = true; m_topicConfiguration = value; }


inline void SetTopicConfiguration(TopicConfigurationDeprecated&& value) { m_topicConfigurationHasBeenSet = true; m_topicConfiguration = std::move(value); }


inline NotificationConfigurationDeprecated& WithTopicConfiguration(const TopicConfigurationDeprecated& value) { SetTopicConfiguration(value); return *this;}


inline NotificationConfigurationDeprecated& WithTopicConfiguration(TopicConfigurationDeprecated&& value) { SetTopicConfiguration(std::move(value)); return *this;}



inline const QueueConfigurationDeprecated& GetQueueConfiguration() const{ return m_queueConfiguration; }


inline void SetQueueConfiguration(const QueueConfigurationDeprecated& value) { m_queueConfigurationHasBeenSet = true; m_queueConfiguration = value; }


inline void SetQueueConfiguration(QueueConfigurationDeprecated&& value) { m_queueConfigurationHasBeenSet = true; m_queueConfiguration = std::move(value); }


inline NotificationConfigurationDeprecated& WithQueueConfiguration(const QueueConfigurationDeprecated& value) { SetQueueConfiguration(value); return *this;}


inline NotificationConfigurationDeprecated& WithQueueConfiguration(QueueConfigurationDeprecated&& value) { SetQueueConfiguration(std::move(value)); return *this;}



inline const CloudFunctionConfiguration& GetCloudFunctionConfiguration() const{ return m_cloudFunctionConfiguration; }


inline void SetCloudFunctionConfiguration(const CloudFunctionConfiguration& value) { m_cloudFunctionConfigurationHasBeenSet = true; m_cloudFunctionConfiguration = value; }


inline void SetCloudFunctionConfiguration(CloudFunctionConfiguration&& value) { m_cloudFunctionConfigurationHasBeenSet = true; m_cloudFunctionConfiguration = std::move(value); }


inline NotificationConfigurationDeprecated& WithCloudFunctionConfiguration(const CloudFunctionConfiguration& value) { SetCloudFunctionConfiguration(value); return *this;}


inline NotificationConfigurationDeprecated& WithCloudFunctionConfiguration(CloudFunctionConfiguration&& value) { SetCloudFunctionConfiguration(std::move(value)); return *this;}

private:

TopicConfigurationDeprecated m_topicConfiguration;
bool m_topicConfigurationHasBeenSet;

QueueConfigurationDeprecated m_queueConfiguration;
bool m_queueConfigurationHasBeenSet;

CloudFunctionConfiguration m_cloudFunctionConfiguration;
bool m_cloudFunctionConfigurationHasBeenSet;
};

} 
} 
} 
