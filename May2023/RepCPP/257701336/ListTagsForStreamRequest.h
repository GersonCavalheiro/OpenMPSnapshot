

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/kinesis/KinesisRequest.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <utility>

namespace Aws
{
namespace Kinesis
{
namespace Model
{


class AWS_KINESIS_API ListTagsForStreamRequest : public KinesisRequest
{
public:
ListTagsForStreamRequest();

inline virtual const char* GetServiceRequestName() const override { return "ListTagsForStream"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetStreamName() const{ return m_streamName; }


inline void SetStreamName(const Aws::String& value) { m_streamNameHasBeenSet = true; m_streamName = value; }


inline void SetStreamName(Aws::String&& value) { m_streamNameHasBeenSet = true; m_streamName = std::move(value); }


inline void SetStreamName(const char* value) { m_streamNameHasBeenSet = true; m_streamName.assign(value); }


inline ListTagsForStreamRequest& WithStreamName(const Aws::String& value) { SetStreamName(value); return *this;}


inline ListTagsForStreamRequest& WithStreamName(Aws::String&& value) { SetStreamName(std::move(value)); return *this;}


inline ListTagsForStreamRequest& WithStreamName(const char* value) { SetStreamName(value); return *this;}



inline const Aws::String& GetExclusiveStartTagKey() const{ return m_exclusiveStartTagKey; }


inline void SetExclusiveStartTagKey(const Aws::String& value) { m_exclusiveStartTagKeyHasBeenSet = true; m_exclusiveStartTagKey = value; }


inline void SetExclusiveStartTagKey(Aws::String&& value) { m_exclusiveStartTagKeyHasBeenSet = true; m_exclusiveStartTagKey = std::move(value); }


inline void SetExclusiveStartTagKey(const char* value) { m_exclusiveStartTagKeyHasBeenSet = true; m_exclusiveStartTagKey.assign(value); }


inline ListTagsForStreamRequest& WithExclusiveStartTagKey(const Aws::String& value) { SetExclusiveStartTagKey(value); return *this;}


inline ListTagsForStreamRequest& WithExclusiveStartTagKey(Aws::String&& value) { SetExclusiveStartTagKey(std::move(value)); return *this;}


inline ListTagsForStreamRequest& WithExclusiveStartTagKey(const char* value) { SetExclusiveStartTagKey(value); return *this;}



inline int GetLimit() const{ return m_limit; }


inline void SetLimit(int value) { m_limitHasBeenSet = true; m_limit = value; }


inline ListTagsForStreamRequest& WithLimit(int value) { SetLimit(value); return *this;}

private:

Aws::String m_streamName;
bool m_streamNameHasBeenSet;

Aws::String m_exclusiveStartTagKey;
bool m_exclusiveStartTagKeyHasBeenSet;

int m_limit;
bool m_limitHasBeenSet;
};

} 
} 
} 
