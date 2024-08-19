

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


class AWS_KINESIS_API MergeShardsRequest : public KinesisRequest
{
public:
MergeShardsRequest();

inline virtual const char* GetServiceRequestName() const override { return "MergeShards"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetStreamName() const{ return m_streamName; }


inline void SetStreamName(const Aws::String& value) { m_streamNameHasBeenSet = true; m_streamName = value; }


inline void SetStreamName(Aws::String&& value) { m_streamNameHasBeenSet = true; m_streamName = std::move(value); }


inline void SetStreamName(const char* value) { m_streamNameHasBeenSet = true; m_streamName.assign(value); }


inline MergeShardsRequest& WithStreamName(const Aws::String& value) { SetStreamName(value); return *this;}


inline MergeShardsRequest& WithStreamName(Aws::String&& value) { SetStreamName(std::move(value)); return *this;}


inline MergeShardsRequest& WithStreamName(const char* value) { SetStreamName(value); return *this;}



inline const Aws::String& GetShardToMerge() const{ return m_shardToMerge; }


inline void SetShardToMerge(const Aws::String& value) { m_shardToMergeHasBeenSet = true; m_shardToMerge = value; }


inline void SetShardToMerge(Aws::String&& value) { m_shardToMergeHasBeenSet = true; m_shardToMerge = std::move(value); }


inline void SetShardToMerge(const char* value) { m_shardToMergeHasBeenSet = true; m_shardToMerge.assign(value); }


inline MergeShardsRequest& WithShardToMerge(const Aws::String& value) { SetShardToMerge(value); return *this;}


inline MergeShardsRequest& WithShardToMerge(Aws::String&& value) { SetShardToMerge(std::move(value)); return *this;}


inline MergeShardsRequest& WithShardToMerge(const char* value) { SetShardToMerge(value); return *this;}



inline const Aws::String& GetAdjacentShardToMerge() const{ return m_adjacentShardToMerge; }


inline void SetAdjacentShardToMerge(const Aws::String& value) { m_adjacentShardToMergeHasBeenSet = true; m_adjacentShardToMerge = value; }


inline void SetAdjacentShardToMerge(Aws::String&& value) { m_adjacentShardToMergeHasBeenSet = true; m_adjacentShardToMerge = std::move(value); }


inline void SetAdjacentShardToMerge(const char* value) { m_adjacentShardToMergeHasBeenSet = true; m_adjacentShardToMerge.assign(value); }


inline MergeShardsRequest& WithAdjacentShardToMerge(const Aws::String& value) { SetAdjacentShardToMerge(value); return *this;}


inline MergeShardsRequest& WithAdjacentShardToMerge(Aws::String&& value) { SetAdjacentShardToMerge(std::move(value)); return *this;}


inline MergeShardsRequest& WithAdjacentShardToMerge(const char* value) { SetAdjacentShardToMerge(value); return *this;}

private:

Aws::String m_streamName;
bool m_streamNameHasBeenSet;

Aws::String m_shardToMerge;
bool m_shardToMergeHasBeenSet;

Aws::String m_adjacentShardToMerge;
bool m_adjacentShardToMergeHasBeenSet;
};

} 
} 
} 
