

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/kinesis/KinesisRequest.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/kinesis/model/ShardIteratorType.h>
#include <aws/core/utils/DateTime.h>
#include <utility>

namespace Aws
{
namespace Kinesis
{
namespace Model
{


class AWS_KINESIS_API GetShardIteratorRequest : public KinesisRequest
{
public:
GetShardIteratorRequest();

inline virtual const char* GetServiceRequestName() const override { return "GetShardIterator"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetStreamName() const{ return m_streamName; }


inline void SetStreamName(const Aws::String& value) { m_streamNameHasBeenSet = true; m_streamName = value; }


inline void SetStreamName(Aws::String&& value) { m_streamNameHasBeenSet = true; m_streamName = std::move(value); }


inline void SetStreamName(const char* value) { m_streamNameHasBeenSet = true; m_streamName.assign(value); }


inline GetShardIteratorRequest& WithStreamName(const Aws::String& value) { SetStreamName(value); return *this;}


inline GetShardIteratorRequest& WithStreamName(Aws::String&& value) { SetStreamName(std::move(value)); return *this;}


inline GetShardIteratorRequest& WithStreamName(const char* value) { SetStreamName(value); return *this;}



inline const Aws::String& GetShardId() const{ return m_shardId; }


inline void SetShardId(const Aws::String& value) { m_shardIdHasBeenSet = true; m_shardId = value; }


inline void SetShardId(Aws::String&& value) { m_shardIdHasBeenSet = true; m_shardId = std::move(value); }


inline void SetShardId(const char* value) { m_shardIdHasBeenSet = true; m_shardId.assign(value); }


inline GetShardIteratorRequest& WithShardId(const Aws::String& value) { SetShardId(value); return *this;}


inline GetShardIteratorRequest& WithShardId(Aws::String&& value) { SetShardId(std::move(value)); return *this;}


inline GetShardIteratorRequest& WithShardId(const char* value) { SetShardId(value); return *this;}



inline const ShardIteratorType& GetShardIteratorType() const{ return m_shardIteratorType; }


inline void SetShardIteratorType(const ShardIteratorType& value) { m_shardIteratorTypeHasBeenSet = true; m_shardIteratorType = value; }


inline void SetShardIteratorType(ShardIteratorType&& value) { m_shardIteratorTypeHasBeenSet = true; m_shardIteratorType = std::move(value); }


inline GetShardIteratorRequest& WithShardIteratorType(const ShardIteratorType& value) { SetShardIteratorType(value); return *this;}


inline GetShardIteratorRequest& WithShardIteratorType(ShardIteratorType&& value) { SetShardIteratorType(std::move(value)); return *this;}



inline const Aws::String& GetStartingSequenceNumber() const{ return m_startingSequenceNumber; }


inline void SetStartingSequenceNumber(const Aws::String& value) { m_startingSequenceNumberHasBeenSet = true; m_startingSequenceNumber = value; }


inline void SetStartingSequenceNumber(Aws::String&& value) { m_startingSequenceNumberHasBeenSet = true; m_startingSequenceNumber = std::move(value); }


inline void SetStartingSequenceNumber(const char* value) { m_startingSequenceNumberHasBeenSet = true; m_startingSequenceNumber.assign(value); }


inline GetShardIteratorRequest& WithStartingSequenceNumber(const Aws::String& value) { SetStartingSequenceNumber(value); return *this;}


inline GetShardIteratorRequest& WithStartingSequenceNumber(Aws::String&& value) { SetStartingSequenceNumber(std::move(value)); return *this;}


inline GetShardIteratorRequest& WithStartingSequenceNumber(const char* value) { SetStartingSequenceNumber(value); return *this;}



inline const Aws::Utils::DateTime& GetTimestamp() const{ return m_timestamp; }


inline void SetTimestamp(const Aws::Utils::DateTime& value) { m_timestampHasBeenSet = true; m_timestamp = value; }


inline void SetTimestamp(Aws::Utils::DateTime&& value) { m_timestampHasBeenSet = true; m_timestamp = std::move(value); }


inline GetShardIteratorRequest& WithTimestamp(const Aws::Utils::DateTime& value) { SetTimestamp(value); return *this;}


inline GetShardIteratorRequest& WithTimestamp(Aws::Utils::DateTime&& value) { SetTimestamp(std::move(value)); return *this;}

private:

Aws::String m_streamName;
bool m_streamNameHasBeenSet;

Aws::String m_shardId;
bool m_shardIdHasBeenSet;

ShardIteratorType m_shardIteratorType;
bool m_shardIteratorTypeHasBeenSet;

Aws::String m_startingSequenceNumber;
bool m_startingSequenceNumberHasBeenSet;

Aws::Utils::DateTime m_timestamp;
bool m_timestampHasBeenSet;
};

} 
} 
} 
