

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/Event.h>
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

class AWS_S3_API TopicConfigurationDeprecated
{
public:
TopicConfigurationDeprecated();
TopicConfigurationDeprecated(const Aws::Utils::Xml::XmlNode& xmlNode);
TopicConfigurationDeprecated& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetId() const{ return m_id; }


inline void SetId(const Aws::String& value) { m_idHasBeenSet = true; m_id = value; }


inline void SetId(Aws::String&& value) { m_idHasBeenSet = true; m_id = std::move(value); }


inline void SetId(const char* value) { m_idHasBeenSet = true; m_id.assign(value); }


inline TopicConfigurationDeprecated& WithId(const Aws::String& value) { SetId(value); return *this;}


inline TopicConfigurationDeprecated& WithId(Aws::String&& value) { SetId(std::move(value)); return *this;}


inline TopicConfigurationDeprecated& WithId(const char* value) { SetId(value); return *this;}



inline const Aws::Vector<Event>& GetEvents() const{ return m_events; }


inline void SetEvents(const Aws::Vector<Event>& value) { m_eventsHasBeenSet = true; m_events = value; }


inline void SetEvents(Aws::Vector<Event>&& value) { m_eventsHasBeenSet = true; m_events = std::move(value); }


inline TopicConfigurationDeprecated& WithEvents(const Aws::Vector<Event>& value) { SetEvents(value); return *this;}


inline TopicConfigurationDeprecated& WithEvents(Aws::Vector<Event>&& value) { SetEvents(std::move(value)); return *this;}


inline TopicConfigurationDeprecated& AddEvents(const Event& value) { m_eventsHasBeenSet = true; m_events.push_back(value); return *this; }


inline TopicConfigurationDeprecated& AddEvents(Event&& value) { m_eventsHasBeenSet = true; m_events.push_back(std::move(value)); return *this; }



inline const Aws::String& GetTopic() const{ return m_topic; }


inline void SetTopic(const Aws::String& value) { m_topicHasBeenSet = true; m_topic = value; }


inline void SetTopic(Aws::String&& value) { m_topicHasBeenSet = true; m_topic = std::move(value); }


inline void SetTopic(const char* value) { m_topicHasBeenSet = true; m_topic.assign(value); }


inline TopicConfigurationDeprecated& WithTopic(const Aws::String& value) { SetTopic(value); return *this;}


inline TopicConfigurationDeprecated& WithTopic(Aws::String&& value) { SetTopic(std::move(value)); return *this;}


inline TopicConfigurationDeprecated& WithTopic(const char* value) { SetTopic(value); return *this;}

private:

Aws::String m_id;
bool m_idHasBeenSet;

Aws::Vector<Event> m_events;
bool m_eventsHasBeenSet;

Aws::String m_topic;
bool m_topicHasBeenSet;
};

} 
} 
} 
