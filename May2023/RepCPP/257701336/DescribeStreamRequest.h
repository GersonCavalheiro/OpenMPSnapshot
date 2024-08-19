

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


class AWS_KINESIS_API DescribeStreamRequest : public KinesisRequest
{
public:
DescribeStreamRequest();

inline virtual const char* GetServiceRequestName() const override { return "DescribeStream"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetStreamName() const{ return m_streamName; }


inline void SetStreamName(const Aws::String& value) { m_streamNameHasBeenSet = true; m_streamName = value; }


inline void SetStreamName(Aws::String&& value) { m_streamNameHasBeenSet = true; m_streamName = std::move(value); }


inline void SetStreamName(const char* value) { m_streamNameHasBeenSet = true; m_streamName.assign(value); }


inline DescribeStreamRequest& WithStreamName(const Aws::String& value) { SetStreamName(value); return *this;}


inline DescribeStreamRequest& WithStreamName(Aws::String&& value) { SetStreamName(std::move(value)); return *this;}


inline DescribeStreamRequest& WithStreamName(const char* value) { SetStreamName(value); return *this;}



inline int GetLimit() const{ return m_limit; }


inline void SetLimit(int value) { m_limitHasBeenSet = true; m_limit = value; }


inline DescribeStreamRequest& WithLimit(int value) { SetLimit(value); return *this;}



inline const Aws::String& GetExclusiveStartShardId() const{ return m_exclusiveStartShardId; }


inline void SetExclusiveStartShardId(const Aws::String& value) { m_exclusiveStartShardIdHasBeenSet = true; m_exclusiveStartShardId = value; }


inline void SetExclusiveStartShardId(Aws::String&& value) { m_exclusiveStartShardIdHasBeenSet = true; m_exclusiveStartShardId = std::move(value); }


inline void SetExclusiveStartShardId(const char* value) { m_exclusiveStartShardIdHasBeenSet = true; m_exclusiveStartShardId.assign(value); }


inline DescribeStreamRequest& WithExclusiveStartShardId(const Aws::String& value) { SetExclusiveStartShardId(value); return *this;}


inline DescribeStreamRequest& WithExclusiveStartShardId(Aws::String&& value) { SetExclusiveStartShardId(std::move(value)); return *this;}


inline DescribeStreamRequest& WithExclusiveStartShardId(const char* value) { SetExclusiveStartShardId(value); return *this;}

private:

Aws::String m_streamName;
bool m_streamNameHasBeenSet;

int m_limit;
bool m_limitHasBeenSet;

Aws::String m_exclusiveStartShardId;
bool m_exclusiveStartShardIdHasBeenSet;
};

} 
} 
} 
