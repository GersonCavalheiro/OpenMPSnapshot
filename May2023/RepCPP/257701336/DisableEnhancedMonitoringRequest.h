

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/kinesis/KinesisRequest.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/kinesis/model/MetricsName.h>
#include <utility>

namespace Aws
{
namespace Kinesis
{
namespace Model
{


class AWS_KINESIS_API DisableEnhancedMonitoringRequest : public KinesisRequest
{
public:
DisableEnhancedMonitoringRequest();

inline virtual const char* GetServiceRequestName() const override { return "DisableEnhancedMonitoring"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetStreamName() const{ return m_streamName; }


inline void SetStreamName(const Aws::String& value) { m_streamNameHasBeenSet = true; m_streamName = value; }


inline void SetStreamName(Aws::String&& value) { m_streamNameHasBeenSet = true; m_streamName = std::move(value); }


inline void SetStreamName(const char* value) { m_streamNameHasBeenSet = true; m_streamName.assign(value); }


inline DisableEnhancedMonitoringRequest& WithStreamName(const Aws::String& value) { SetStreamName(value); return *this;}


inline DisableEnhancedMonitoringRequest& WithStreamName(Aws::String&& value) { SetStreamName(std::move(value)); return *this;}


inline DisableEnhancedMonitoringRequest& WithStreamName(const char* value) { SetStreamName(value); return *this;}



inline const Aws::Vector<MetricsName>& GetShardLevelMetrics() const{ return m_shardLevelMetrics; }


inline void SetShardLevelMetrics(const Aws::Vector<MetricsName>& value) { m_shardLevelMetricsHasBeenSet = true; m_shardLevelMetrics = value; }


inline void SetShardLevelMetrics(Aws::Vector<MetricsName>&& value) { m_shardLevelMetricsHasBeenSet = true; m_shardLevelMetrics = std::move(value); }


inline DisableEnhancedMonitoringRequest& WithShardLevelMetrics(const Aws::Vector<MetricsName>& value) { SetShardLevelMetrics(value); return *this;}


inline DisableEnhancedMonitoringRequest& WithShardLevelMetrics(Aws::Vector<MetricsName>&& value) { SetShardLevelMetrics(std::move(value)); return *this;}


inline DisableEnhancedMonitoringRequest& AddShardLevelMetrics(const MetricsName& value) { m_shardLevelMetricsHasBeenSet = true; m_shardLevelMetrics.push_back(value); return *this; }


inline DisableEnhancedMonitoringRequest& AddShardLevelMetrics(MetricsName&& value) { m_shardLevelMetricsHasBeenSet = true; m_shardLevelMetrics.push_back(std::move(value)); return *this; }

private:

Aws::String m_streamName;
bool m_streamNameHasBeenSet;

Aws::Vector<MetricsName> m_shardLevelMetrics;
bool m_shardLevelMetricsHasBeenSet;
};

} 
} 
} 
