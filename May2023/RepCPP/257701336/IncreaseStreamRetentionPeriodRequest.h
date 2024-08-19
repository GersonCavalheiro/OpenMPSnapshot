

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


class AWS_KINESIS_API IncreaseStreamRetentionPeriodRequest : public KinesisRequest
{
public:
IncreaseStreamRetentionPeriodRequest();

inline virtual const char* GetServiceRequestName() const override { return "IncreaseStreamRetentionPeriod"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetStreamName() const{ return m_streamName; }


inline void SetStreamName(const Aws::String& value) { m_streamNameHasBeenSet = true; m_streamName = value; }


inline void SetStreamName(Aws::String&& value) { m_streamNameHasBeenSet = true; m_streamName = std::move(value); }


inline void SetStreamName(const char* value) { m_streamNameHasBeenSet = true; m_streamName.assign(value); }


inline IncreaseStreamRetentionPeriodRequest& WithStreamName(const Aws::String& value) { SetStreamName(value); return *this;}


inline IncreaseStreamRetentionPeriodRequest& WithStreamName(Aws::String&& value) { SetStreamName(std::move(value)); return *this;}


inline IncreaseStreamRetentionPeriodRequest& WithStreamName(const char* value) { SetStreamName(value); return *this;}



inline int GetRetentionPeriodHours() const{ return m_retentionPeriodHours; }


inline void SetRetentionPeriodHours(int value) { m_retentionPeriodHoursHasBeenSet = true; m_retentionPeriodHours = value; }


inline IncreaseStreamRetentionPeriodRequest& WithRetentionPeriodHours(int value) { SetRetentionPeriodHours(value); return *this;}

private:

Aws::String m_streamName;
bool m_streamNameHasBeenSet;

int m_retentionPeriodHours;
bool m_retentionPeriodHoursHasBeenSet;
};

} 
} 
} 
