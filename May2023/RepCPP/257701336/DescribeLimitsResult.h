

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>

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
class AWS_KINESIS_API DescribeLimitsResult
{
public:
DescribeLimitsResult();
DescribeLimitsResult(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);
DescribeLimitsResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);



inline int GetShardLimit() const{ return m_shardLimit; }


inline void SetShardLimit(int value) { m_shardLimit = value; }


inline DescribeLimitsResult& WithShardLimit(int value) { SetShardLimit(value); return *this;}



inline int GetOpenShardCount() const{ return m_openShardCount; }


inline void SetOpenShardCount(int value) { m_openShardCount = value; }


inline DescribeLimitsResult& WithOpenShardCount(int value) { SetOpenShardCount(value); return *this;}

private:

int m_shardLimit;

int m_openShardCount;
};

} 
} 
} 
