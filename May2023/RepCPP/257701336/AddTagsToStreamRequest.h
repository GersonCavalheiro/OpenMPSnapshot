

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/kinesis/KinesisRequest.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSMap.h>
#include <utility>

namespace Aws
{
namespace Kinesis
{
namespace Model
{


class AWS_KINESIS_API AddTagsToStreamRequest : public KinesisRequest
{
public:
AddTagsToStreamRequest();

inline virtual const char* GetServiceRequestName() const override { return "AddTagsToStream"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetStreamName() const{ return m_streamName; }


inline void SetStreamName(const Aws::String& value) { m_streamNameHasBeenSet = true; m_streamName = value; }


inline void SetStreamName(Aws::String&& value) { m_streamNameHasBeenSet = true; m_streamName = std::move(value); }


inline void SetStreamName(const char* value) { m_streamNameHasBeenSet = true; m_streamName.assign(value); }


inline AddTagsToStreamRequest& WithStreamName(const Aws::String& value) { SetStreamName(value); return *this;}


inline AddTagsToStreamRequest& WithStreamName(Aws::String&& value) { SetStreamName(std::move(value)); return *this;}


inline AddTagsToStreamRequest& WithStreamName(const char* value) { SetStreamName(value); return *this;}



inline const Aws::Map<Aws::String, Aws::String>& GetTags() const{ return m_tags; }


inline void SetTags(const Aws::Map<Aws::String, Aws::String>& value) { m_tagsHasBeenSet = true; m_tags = value; }


inline void SetTags(Aws::Map<Aws::String, Aws::String>&& value) { m_tagsHasBeenSet = true; m_tags = std::move(value); }


inline AddTagsToStreamRequest& WithTags(const Aws::Map<Aws::String, Aws::String>& value) { SetTags(value); return *this;}


inline AddTagsToStreamRequest& WithTags(Aws::Map<Aws::String, Aws::String>&& value) { SetTags(std::move(value)); return *this;}


inline AddTagsToStreamRequest& AddTags(const Aws::String& key, const Aws::String& value) { m_tagsHasBeenSet = true; m_tags.emplace(key, value); return *this; }


inline AddTagsToStreamRequest& AddTags(Aws::String&& key, const Aws::String& value) { m_tagsHasBeenSet = true; m_tags.emplace(std::move(key), value); return *this; }


inline AddTagsToStreamRequest& AddTags(const Aws::String& key, Aws::String&& value) { m_tagsHasBeenSet = true; m_tags.emplace(key, std::move(value)); return *this; }


inline AddTagsToStreamRequest& AddTags(Aws::String&& key, Aws::String&& value) { m_tagsHasBeenSet = true; m_tags.emplace(std::move(key), std::move(value)); return *this; }


inline AddTagsToStreamRequest& AddTags(const char* key, Aws::String&& value) { m_tagsHasBeenSet = true; m_tags.emplace(key, std::move(value)); return *this; }


inline AddTagsToStreamRequest& AddTags(Aws::String&& key, const char* value) { m_tagsHasBeenSet = true; m_tags.emplace(std::move(key), value); return *this; }


inline AddTagsToStreamRequest& AddTags(const char* key, const char* value) { m_tagsHasBeenSet = true; m_tags.emplace(key, value); return *this; }

private:

Aws::String m_streamName;
bool m_streamNameHasBeenSet;

Aws::Map<Aws::String, Aws::String> m_tags;
bool m_tagsHasBeenSet;
};

} 
} 
} 
