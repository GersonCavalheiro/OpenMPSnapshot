

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
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
class AWS_KINESIS_API UpdateShardCountResult
{
public:
UpdateShardCountResult();
UpdateShardCountResult(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);
UpdateShardCountResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);



inline const Aws::String& GetStreamName() const{ return m_streamName; }


inline void SetStreamName(const Aws::String& value) { m_streamName = value; }


inline void SetStreamName(Aws::String&& value) { m_streamName = std::move(value); }


inline void SetStreamName(const char* value) { m_streamName.assign(value); }


inline UpdateShardCountResult& WithStreamName(const Aws::String& value) { SetStreamName(value); return *this;}


inline UpdateShardCountResult& WithStreamName(Aws::String&& value) { SetStreamName(std::move(value)); return *this;}


inline UpdateShardCountResult& WithStreamName(const char* value) { SetStreamName(value); return *this;}



inline int GetCurrentShardCount() const{ return m_currentShardCount; }


inline void SetCurrentShardCount(int value) { m_currentShardCount = value; }


inline UpdateShardCountResult& WithCurrentShardCount(int value) { SetCurrentShardCount(value); return *this;}



inline int GetTargetShardCount() const{ return m_targetShardCount; }


inline void SetTargetShardCount(int value) { m_targetShardCount = value; }


inline UpdateShardCountResult& WithTargetShardCount(int value) { SetTargetShardCount(value); return *this;}

private:

Aws::String m_streamName;

int m_currentShardCount;

int m_targetShardCount;
};

} 
} 
} 
