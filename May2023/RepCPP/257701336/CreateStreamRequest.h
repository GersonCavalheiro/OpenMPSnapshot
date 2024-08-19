

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


class AWS_KINESIS_API CreateStreamRequest : public KinesisRequest
{
public:
CreateStreamRequest();

inline virtual const char* GetServiceRequestName() const override { return "CreateStream"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetStreamName() const{ return m_streamName; }


inline void SetStreamName(const Aws::String& value) { m_streamNameHasBeenSet = true; m_streamName = value; }


inline void SetStreamName(Aws::String&& value) { m_streamNameHasBeenSet = true; m_streamName = std::move(value); }


inline void SetStreamName(const char* value) { m_streamNameHasBeenSet = true; m_streamName.assign(value); }


inline CreateStreamRequest& WithStreamName(const Aws::String& value) { SetStreamName(value); return *this;}


inline CreateStreamRequest& WithStreamName(Aws::String&& value) { SetStreamName(std::move(value)); return *this;}


inline CreateStreamRequest& WithStreamName(const char* value) { SetStreamName(value); return *this;}



inline int GetShardCount() const{ return m_shardCount; }


inline void SetShardCount(int value) { m_shardCountHasBeenSet = true; m_shardCount = value; }


inline CreateStreamRequest& WithShardCount(int value) { SetShardCount(value); return *this;}

private:

Aws::String m_streamName;
bool m_streamNameHasBeenSet;

int m_shardCount;
bool m_shardCountHasBeenSet;
};

} 
} 
} 
