

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


class AWS_KINESIS_API SplitShardRequest : public KinesisRequest
{
public:
SplitShardRequest();

inline virtual const char* GetServiceRequestName() const override { return "SplitShard"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetStreamName() const{ return m_streamName; }


inline void SetStreamName(const Aws::String& value) { m_streamNameHasBeenSet = true; m_streamName = value; }


inline void SetStreamName(Aws::String&& value) { m_streamNameHasBeenSet = true; m_streamName = std::move(value); }


inline void SetStreamName(const char* value) { m_streamNameHasBeenSet = true; m_streamName.assign(value); }


inline SplitShardRequest& WithStreamName(const Aws::String& value) { SetStreamName(value); return *this;}


inline SplitShardRequest& WithStreamName(Aws::String&& value) { SetStreamName(std::move(value)); return *this;}


inline SplitShardRequest& WithStreamName(const char* value) { SetStreamName(value); return *this;}



inline const Aws::String& GetShardToSplit() const{ return m_shardToSplit; }


inline void SetShardToSplit(const Aws::String& value) { m_shardToSplitHasBeenSet = true; m_shardToSplit = value; }


inline void SetShardToSplit(Aws::String&& value) { m_shardToSplitHasBeenSet = true; m_shardToSplit = std::move(value); }


inline void SetShardToSplit(const char* value) { m_shardToSplitHasBeenSet = true; m_shardToSplit.assign(value); }


inline SplitShardRequest& WithShardToSplit(const Aws::String& value) { SetShardToSplit(value); return *this;}


inline SplitShardRequest& WithShardToSplit(Aws::String&& value) { SetShardToSplit(std::move(value)); return *this;}


inline SplitShardRequest& WithShardToSplit(const char* value) { SetShardToSplit(value); return *this;}



inline const Aws::String& GetNewStartingHashKey() const{ return m_newStartingHashKey; }


inline void SetNewStartingHashKey(const Aws::String& value) { m_newStartingHashKeyHasBeenSet = true; m_newStartingHashKey = value; }


inline void SetNewStartingHashKey(Aws::String&& value) { m_newStartingHashKeyHasBeenSet = true; m_newStartingHashKey = std::move(value); }


inline void SetNewStartingHashKey(const char* value) { m_newStartingHashKeyHasBeenSet = true; m_newStartingHashKey.assign(value); }


inline SplitShardRequest& WithNewStartingHashKey(const Aws::String& value) { SetNewStartingHashKey(value); return *this;}


inline SplitShardRequest& WithNewStartingHashKey(Aws::String&& value) { SetNewStartingHashKey(std::move(value)); return *this;}


inline SplitShardRequest& WithNewStartingHashKey(const char* value) { SetNewStartingHashKey(value); return *this;}

private:

Aws::String m_streamName;
bool m_streamNameHasBeenSet;

Aws::String m_shardToSplit;
bool m_shardToSplitHasBeenSet;

Aws::String m_newStartingHashKey;
bool m_newStartingHashKeyHasBeenSet;
};

} 
} 
} 
