

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/kinesis/KinesisRequest.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <utility>

namespace Aws
{
namespace Kinesis
{
namespace Model
{


class AWS_KINESIS_API RemoveTagsFromStreamRequest : public KinesisRequest
{
public:
RemoveTagsFromStreamRequest();

inline virtual const char* GetServiceRequestName() const override { return "RemoveTagsFromStream"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetStreamName() const{ return m_streamName; }


inline void SetStreamName(const Aws::String& value) { m_streamNameHasBeenSet = true; m_streamName = value; }


inline void SetStreamName(Aws::String&& value) { m_streamNameHasBeenSet = true; m_streamName = std::move(value); }


inline void SetStreamName(const char* value) { m_streamNameHasBeenSet = true; m_streamName.assign(value); }


inline RemoveTagsFromStreamRequest& WithStreamName(const Aws::String& value) { SetStreamName(value); return *this;}


inline RemoveTagsFromStreamRequest& WithStreamName(Aws::String&& value) { SetStreamName(std::move(value)); return *this;}


inline RemoveTagsFromStreamRequest& WithStreamName(const char* value) { SetStreamName(value); return *this;}



inline const Aws::Vector<Aws::String>& GetTagKeys() const{ return m_tagKeys; }


inline void SetTagKeys(const Aws::Vector<Aws::String>& value) { m_tagKeysHasBeenSet = true; m_tagKeys = value; }


inline void SetTagKeys(Aws::Vector<Aws::String>&& value) { m_tagKeysHasBeenSet = true; m_tagKeys = std::move(value); }


inline RemoveTagsFromStreamRequest& WithTagKeys(const Aws::Vector<Aws::String>& value) { SetTagKeys(value); return *this;}


inline RemoveTagsFromStreamRequest& WithTagKeys(Aws::Vector<Aws::String>&& value) { SetTagKeys(std::move(value)); return *this;}


inline RemoveTagsFromStreamRequest& AddTagKeys(const Aws::String& value) { m_tagKeysHasBeenSet = true; m_tagKeys.push_back(value); return *this; }


inline RemoveTagsFromStreamRequest& AddTagKeys(Aws::String&& value) { m_tagKeysHasBeenSet = true; m_tagKeys.push_back(std::move(value)); return *this; }


inline RemoveTagsFromStreamRequest& AddTagKeys(const char* value) { m_tagKeysHasBeenSet = true; m_tagKeys.push_back(value); return *this; }

private:

Aws::String m_streamName;
bool m_streamNameHasBeenSet;

Aws::Vector<Aws::String> m_tagKeys;
bool m_tagKeysHasBeenSet;
};

} 
} 
} 
