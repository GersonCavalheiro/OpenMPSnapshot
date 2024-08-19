

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/TopicConfiguration.h>
#include <aws/s3/model/QueueConfiguration.h>
#include <aws/s3/model/LambdaFunctionConfiguration.h>
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


class AWS_S3_API NotificationConfiguration
{
public:
NotificationConfiguration();
NotificationConfiguration(const Aws::Utils::Xml::XmlNode& xmlNode);
NotificationConfiguration& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::Vector<TopicConfiguration>& GetTopicConfigurations() const{ return m_topicConfigurations; }


inline void SetTopicConfigurations(const Aws::Vector<TopicConfiguration>& value) { m_topicConfigurationsHasBeenSet = true; m_topicConfigurations = value; }


inline void SetTopicConfigurations(Aws::Vector<TopicConfiguration>&& value) { m_topicConfigurationsHasBeenSet = true; m_topicConfigurations = std::move(value); }


inline NotificationConfiguration& WithTopicConfigurations(const Aws::Vector<TopicConfiguration>& value) { SetTopicConfigurations(value); return *this;}


inline NotificationConfiguration& WithTopicConfigurations(Aws::Vector<TopicConfiguration>&& value) { SetTopicConfigurations(std::move(value)); return *this;}


inline NotificationConfiguration& AddTopicConfigurations(const TopicConfiguration& value) { m_topicConfigurationsHasBeenSet = true; m_topicConfigurations.push_back(value); return *this; }


inline NotificationConfiguration& AddTopicConfigurations(TopicConfiguration&& value) { m_topicConfigurationsHasBeenSet = true; m_topicConfigurations.push_back(std::move(value)); return *this; }



inline const Aws::Vector<QueueConfiguration>& GetQueueConfigurations() const{ return m_queueConfigurations; }


inline void SetQueueConfigurations(const Aws::Vector<QueueConfiguration>& value) { m_queueConfigurationsHasBeenSet = true; m_queueConfigurations = value; }


inline void SetQueueConfigurations(Aws::Vector<QueueConfiguration>&& value) { m_queueConfigurationsHasBeenSet = true; m_queueConfigurations = std::move(value); }


inline NotificationConfiguration& WithQueueConfigurations(const Aws::Vector<QueueConfiguration>& value) { SetQueueConfigurations(value); return *this;}


inline NotificationConfiguration& WithQueueConfigurations(Aws::Vector<QueueConfiguration>&& value) { SetQueueConfigurations(std::move(value)); return *this;}


inline NotificationConfiguration& AddQueueConfigurations(const QueueConfiguration& value) { m_queueConfigurationsHasBeenSet = true; m_queueConfigurations.push_back(value); return *this; }


inline NotificationConfiguration& AddQueueConfigurations(QueueConfiguration&& value) { m_queueConfigurationsHasBeenSet = true; m_queueConfigurations.push_back(std::move(value)); return *this; }



inline const Aws::Vector<LambdaFunctionConfiguration>& GetLambdaFunctionConfigurations() const{ return m_lambdaFunctionConfigurations; }


inline void SetLambdaFunctionConfigurations(const Aws::Vector<LambdaFunctionConfiguration>& value) { m_lambdaFunctionConfigurationsHasBeenSet = true; m_lambdaFunctionConfigurations = value; }


inline void SetLambdaFunctionConfigurations(Aws::Vector<LambdaFunctionConfiguration>&& value) { m_lambdaFunctionConfigurationsHasBeenSet = true; m_lambdaFunctionConfigurations = std::move(value); }


inline NotificationConfiguration& WithLambdaFunctionConfigurations(const Aws::Vector<LambdaFunctionConfiguration>& value) { SetLambdaFunctionConfigurations(value); return *this;}


inline NotificationConfiguration& WithLambdaFunctionConfigurations(Aws::Vector<LambdaFunctionConfiguration>&& value) { SetLambdaFunctionConfigurations(std::move(value)); return *this;}


inline NotificationConfiguration& AddLambdaFunctionConfigurations(const LambdaFunctionConfiguration& value) { m_lambdaFunctionConfigurationsHasBeenSet = true; m_lambdaFunctionConfigurations.push_back(value); return *this; }


inline NotificationConfiguration& AddLambdaFunctionConfigurations(LambdaFunctionConfiguration&& value) { m_lambdaFunctionConfigurationsHasBeenSet = true; m_lambdaFunctionConfigurations.push_back(std::move(value)); return *this; }

private:

Aws::Vector<TopicConfiguration> m_topicConfigurations;
bool m_topicConfigurationsHasBeenSet;

Aws::Vector<QueueConfiguration> m_queueConfigurations;
bool m_queueConfigurationsHasBeenSet;

Aws::Vector<LambdaFunctionConfiguration> m_lambdaFunctionConfigurations;
bool m_lambdaFunctionConfigurationsHasBeenSet;
};

} 
} 
} 
