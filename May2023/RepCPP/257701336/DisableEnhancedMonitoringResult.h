

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/kinesis/model/MetricsName.h>
#include <utility>

namespace Aws
{
template<typename RESULT_TYPE>
class AmazonWebServiceResult;

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

class AWS_KINESIS_API DisableEnhancedMonitoringResult
{
public:
DisableEnhancedMonitoringResult();
DisableEnhancedMonitoringResult(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);
DisableEnhancedMonitoringResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);



inline const Aws::String& GetStreamName() const{ return m_streamName; }


inline void SetStreamName(const Aws::String& value) { m_streamName = value; }


inline void SetStreamName(Aws::String&& value) { m_streamName = std::move(value); }


inline void SetStreamName(const char* value) { m_streamName.assign(value); }


inline DisableEnhancedMonitoringResult& WithStreamName(const Aws::String& value) { SetStreamName(value); return *this;}


inline DisableEnhancedMonitoringResult& WithStreamName(Aws::String&& value) { SetStreamName(std::move(value)); return *this;}


inline DisableEnhancedMonitoringResult& WithStreamName(const char* value) { SetStreamName(value); return *this;}



inline const Aws::Vector<MetricsName>& GetCurrentShardLevelMetrics() const{ return m_currentShardLevelMetrics; }


inline void SetCurrentShardLevelMetrics(const Aws::Vector<MetricsName>& value) { m_currentShardLevelMetrics = value; }


inline void SetCurrentShardLevelMetrics(Aws::Vector<MetricsName>&& value) { m_currentShardLevelMetrics = std::move(value); }


inline DisableEnhancedMonitoringResult& WithCurrentShardLevelMetrics(const Aws::Vector<MetricsName>& value) { SetCurrentShardLevelMetrics(value); return *this;}


inline DisableEnhancedMonitoringResult& WithCurrentShardLevelMetrics(Aws::Vector<MetricsName>&& value) { SetCurrentShardLevelMetrics(std::move(value)); return *this;}


inline DisableEnhancedMonitoringResult& AddCurrentShardLevelMetrics(const MetricsName& value) { m_currentShardLevelMetrics.push_back(value); return *this; }


inline DisableEnhancedMonitoringResult& AddCurrentShardLevelMetrics(MetricsName&& value) { m_currentShardLevelMetrics.push_back(std::move(value)); return *this; }



inline const Aws::Vector<MetricsName>& GetDesiredShardLevelMetrics() const{ return m_desiredShardLevelMetrics; }


inline void SetDesiredShardLevelMetrics(const Aws::Vector<MetricsName>& value) { m_desiredShardLevelMetrics = value; }


inline void SetDesiredShardLevelMetrics(Aws::Vector<MetricsName>&& value) { m_desiredShardLevelMetrics = std::move(value); }


inline DisableEnhancedMonitoringResult& WithDesiredShardLevelMetrics(const Aws::Vector<MetricsName>& value) { SetDesiredShardLevelMetrics(value); return *this;}


inline DisableEnhancedMonitoringResult& WithDesiredShardLevelMetrics(Aws::Vector<MetricsName>&& value) { SetDesiredShardLevelMetrics(std::move(value)); return *this;}


inline DisableEnhancedMonitoringResult& AddDesiredShardLevelMetrics(const MetricsName& value) { m_desiredShardLevelMetrics.push_back(value); return *this; }


inline DisableEnhancedMonitoringResult& AddDesiredShardLevelMetrics(MetricsName&& value) { m_desiredShardLevelMetrics.push_back(std::move(value)); return *this; }

private:

Aws::String m_streamName;

Aws::Vector<MetricsName> m_currentShardLevelMetrics;

Aws::Vector<MetricsName> m_desiredShardLevelMetrics;
};

} 
} 
} 
