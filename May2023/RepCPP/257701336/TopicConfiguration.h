

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/NotificationConfigurationFilter.h>
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


class AWS_S3_API TopicConfiguration
{
public:
TopicConfiguration();
TopicConfiguration(const Aws::Utils::Xml::XmlNode& xmlNode);
TopicConfiguration& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetId() const{ return m_id; }


inline void SetId(const Aws::String& value) { m_idHasBeenSet = true; m_id = value; }


inline void SetId(Aws::String&& value) { m_idHasBeenSet = true; m_id = std::move(value); }


inline void SetId(const char* value) { m_idHasBeenSet = true; m_id.assign(value); }


inline TopicConfiguration& WithId(const Aws::String& value) { SetId(value); return *this;}


inline TopicConfiguration& WithId(Aws::String&& value) { SetId(std::move(value)); return *this;}


inline TopicConfiguration& WithId(const char* value) { SetId(value); return *this;}



inline const Aws::String& GetTopicArn() const{ return m_topicArn; }


inline void SetTopicArn(const Aws::String& value) { m_topicArnHasBeenSet = true; m_topicArn = value; }


inline void SetTopicArn(Aws::String&& value) { m_topicArnHasBeenSet = true; m_topicArn = std::move(value); }


inline void SetTopicArn(const char* value) { m_topicArnHasBeenSet = true; m_topicArn.assign(value); }


inline TopicConfiguration& WithTopicArn(const Aws::String& value) { SetTopicArn(value); return *this;}


inline TopicConfiguration& WithTopicArn(Aws::String&& value) { SetTopicArn(std::move(value)); return *this;}


inline TopicConfiguration& WithTopicArn(const char* value) { SetTopicArn(value); return *this;}



inline const Aws::Vector<Event>& GetEvents() const{ return m_events; }


inline void SetEvents(const Aws::Vector<Event>& value) { m_eventsHasBeenSet = true; m_events = value; }


inline void SetEvents(Aws::Vector<Event>&& value) { m_eventsHasBeenSet = true; m_events = std::move(value); }


inline TopicConfiguration& WithEvents(const Aws::Vector<Event>& value) { SetEvents(value); return *this;}


inline TopicConfiguration& WithEvents(Aws::Vector<Event>&& value) { SetEvents(std::move(value)); return *this;}


inline TopicConfiguration& AddEvents(const Event& value) { m_eventsHasBeenSet = true; m_events.push_back(value); return *this; }


inline TopicConfiguration& AddEvents(Event&& value) { m_eventsHasBeenSet = true; m_events.push_back(std::move(value)); return *this; }



inline const NotificationConfigurationFilter& GetFilter() const{ return m_filter; }


inline void SetFilter(const NotificationConfigurationFilter& value) { m_filterHasBeenSet = true; m_filter = value; }


inline void SetFilter(NotificationConfigurationFilter&& value) { m_filterHasBeenSet = true; m_filter = std::move(value); }


inline TopicConfiguration& WithFilter(const NotificationConfigurationFilter& value) { SetFilter(value); return *this;}


inline TopicConfiguration& WithFilter(NotificationConfigurationFilter&& value) { SetFilter(std::move(value)); return *this;}

private:

Aws::String m_id;
bool m_idHasBeenSet;

Aws::String m_topicArn;
bool m_topicArnHasBeenSet;

Aws::Vector<Event> m_events;
bool m_eventsHasBeenSet;

NotificationConfigurationFilter m_filter;
bool m_filterHasBeenSet;
};

} 
} 
} 
