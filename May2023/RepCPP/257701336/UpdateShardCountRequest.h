

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/kinesis/KinesisRequest.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/kinesis/model/ScalingType.h>
#include <utility>

namespace Aws
{
namespace Kinesis
{
namespace Model
{


class AWS_KINESIS_API UpdateShardCountRequest : public KinesisRequest
{
public:
UpdateShardCountRequest();

inline virtual const char* GetServiceRequestName() const override { return "UpdateShardCount"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetStreamName() const{ return m_streamName; }


inline void SetStreamName(const Aws::String& value) { m_streamNameHasBeenSet = true; m_streamName = value; }


inline void SetStreamName(Aws::String&& value) { m_streamNameHasBeenSet = true; m_streamName = std::move(value); }


inline void SetStreamName(const char* value) { m_streamNameHasBeenSet = true; m_streamName.assign(value); }


inline UpdateShardCountRequest& WithStreamName(const Aws::String& value) { SetStreamName(value); return *this;}


inline UpdateShardCountRequest& WithStreamName(Aws::String&& value) { SetStreamName(std::move(value)); return *this;}


inline UpdateShardCountRequest& WithStreamName(const char* value) { SetStreamName(value); return *this;}



inline int GetTargetShardCount() const{ return m_targetShardCount; }


inline void SetTargetShardCount(int value) { m_targetShardCountHasBeenSet = true; m_targetShardCount = value; }


inline UpdateShardCountRequest& WithTargetShardCount(int value) { SetTargetShardCount(value); return *this;}



inline const ScalingType& GetScalingType() const{ return m_scalingType; }


inline void SetScalingType(const ScalingType& value) { m_scalingTypeHasBeenSet = true; m_scalingType = value; }


inline void SetScalingType(ScalingType&& value) { m_scalingTypeHasBeenSet = true; m_scalingType = std::move(value); }


inline UpdateShardCountRequest& WithScalingType(const ScalingType& value) { SetScalingType(value); return *this;}


inline UpdateShardCountRequest& WithScalingType(ScalingType&& value) { SetScalingType(std::move(value)); return *this;}

private:

Aws::String m_streamName;
bool m_streamNameHasBeenSet;

int m_targetShardCount;
bool m_targetShardCountHasBeenSet;

ScalingType m_scalingType;
bool m_scalingTypeHasBeenSet;
};

} 
} 
} 
