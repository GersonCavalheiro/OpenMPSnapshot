

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/kinesis/model/MetricsName.h>
#include <utility>

namespace Aws
{
namespace Utils
{
namespace Json
{
class JsonValue;
} 
} 
namespace Kinesis
{
namespace Model
{


class AWS_KINESIS_API EnhancedMetrics
{
public:
EnhancedMetrics();
EnhancedMetrics(const Aws::Utils::Json::JsonValue& jsonValue);
EnhancedMetrics& operator=(const Aws::Utils::Json::JsonValue& jsonValue);
Aws::Utils::Json::JsonValue Jsonize() const;



inline const Aws::Vector<MetricsName>& GetShardLevelMetrics() const{ return m_shardLevelMetrics; }


inline void SetShardLevelMetrics(const Aws::Vector<MetricsName>& value) { m_shardLevelMetricsHasBeenSet = true; m_shardLevelMetrics = value; }


inline void SetShardLevelMetrics(Aws::Vector<MetricsName>&& value) { m_shardLevelMetricsHasBeenSet = true; m_shardLevelMetrics = std::move(value); }


inline EnhancedMetrics& WithShardLevelMetrics(const Aws::Vector<MetricsName>& value) { SetShardLevelMetrics(value); return *this;}


inline EnhancedMetrics& WithShardLevelMetrics(Aws::Vector<MetricsName>&& value) { SetShardLevelMetrics(std::move(value)); return *this;}


inline EnhancedMetrics& AddShardLevelMetrics(const MetricsName& value) { m_shardLevelMetricsHasBeenSet = true; m_shardLevelMetrics.push_back(value); return *this; }


inline EnhancedMetrics& AddShardLevelMetrics(MetricsName&& value) { m_shardLevelMetricsHasBeenSet = true; m_shardLevelMetrics.push_back(std::move(value)); return *this; }

private:

Aws::Vector<MetricsName> m_shardLevelMetrics;
bool m_shardLevelMetricsHasBeenSet;
};

} 
} 
} 
