

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/TopicConfiguration.h>
#include <aws/s3/model/QueueConfiguration.h>
#include <aws/s3/model/LambdaFunctionConfiguration.h>
#include <utility>

namespace Aws
{
template<typename RESULT_TYPE>
class AmazonWebServiceResult;

namespace Utils
{
namespace Xml
{
class XmlDocument;
} 
} 
namespace S3
{
namespace Model
{

class AWS_S3_API GetBucketNotificationConfigurationResult
{
public:
GetBucketNotificationConfigurationResult();
GetBucketNotificationConfigurationResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
GetBucketNotificationConfigurationResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const Aws::Vector<TopicConfiguration>& GetTopicConfigurations() const{ return m_topicConfigurations; }


inline void SetTopicConfigurations(const Aws::Vector<TopicConfiguration>& value) { m_topicConfigurations = value; }


inline void SetTopicConfigurations(Aws::Vector<TopicConfiguration>&& value) { m_topicConfigurations = std::move(value); }


inline GetBucketNotificationConfigurationResult& WithTopicConfigurations(const Aws::Vector<TopicConfiguration>& value) { SetTopicConfigurations(value); return *this;}


inline GetBucketNotificationConfigurationResult& WithTopicConfigurations(Aws::Vector<TopicConfiguration>&& value) { SetTopicConfigurations(std::move(value)); return *this;}


inline GetBucketNotificationConfigurationResult& AddTopicConfigurations(const TopicConfiguration& value) { m_topicConfigurations.push_back(value); return *this; }


inline GetBucketNotificationConfigurationResult& AddTopicConfigurations(TopicConfiguration&& value) { m_topicConfigurations.push_back(std::move(value)); return *this; }



inline const Aws::Vector<QueueConfiguration>& GetQueueConfigurations() const{ return m_queueConfigurations; }


inline void SetQueueConfigurations(const Aws::Vector<QueueConfiguration>& value) { m_queueConfigurations = value; }


inline void SetQueueConfigurations(Aws::Vector<QueueConfiguration>&& value) { m_queueConfigurations = std::move(value); }


inline GetBucketNotificationConfigurationResult& WithQueueConfigurations(const Aws::Vector<QueueConfiguration>& value) { SetQueueConfigurations(value); return *this;}


inline GetBucketNotificationConfigurationResult& WithQueueConfigurations(Aws::Vector<QueueConfiguration>&& value) { SetQueueConfigurations(std::move(value)); return *this;}


inline GetBucketNotificationConfigurationResult& AddQueueConfigurations(const QueueConfiguration& value) { m_queueConfigurations.push_back(value); return *this; }


inline GetBucketNotificationConfigurationResult& AddQueueConfigurations(QueueConfiguration&& value) { m_queueConfigurations.push_back(std::move(value)); return *this; }



inline const Aws::Vector<LambdaFunctionConfiguration>& GetLambdaFunctionConfigurations() const{ return m_lambdaFunctionConfigurations; }


inline void SetLambdaFunctionConfigurations(const Aws::Vector<LambdaFunctionConfiguration>& value) { m_lambdaFunctionConfigurations = value; }


inline void SetLambdaFunctionConfigurations(Aws::Vector<LambdaFunctionConfiguration>&& value) { m_lambdaFunctionConfigurations = std::move(value); }


inline GetBucketNotificationConfigurationResult& WithLambdaFunctionConfigurations(const Aws::Vector<LambdaFunctionConfiguration>& value) { SetLambdaFunctionConfigurations(value); return *this;}


inline GetBucketNotificationConfigurationResult& WithLambdaFunctionConfigurations(Aws::Vector<LambdaFunctionConfiguration>&& value) { SetLambdaFunctionConfigurations(std::move(value)); return *this;}


inline GetBucketNotificationConfigurationResult& AddLambdaFunctionConfigurations(const LambdaFunctionConfiguration& value) { m_lambdaFunctionConfigurations.push_back(value); return *this; }


inline GetBucketNotificationConfigurationResult& AddLambdaFunctionConfigurations(LambdaFunctionConfiguration&& value) { m_lambdaFunctionConfigurations.push_back(std::move(value)); return *this; }

private:

Aws::Vector<TopicConfiguration> m_topicConfigurations;

Aws::Vector<QueueConfiguration> m_queueConfigurations;

Aws::Vector<LambdaFunctionConfiguration> m_lambdaFunctionConfigurations;
};

} 
} 
} 
