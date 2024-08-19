

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

class AWS_KINESIS_API GetShardIteratorResult
{
public:
GetShardIteratorResult();
GetShardIteratorResult(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);
GetShardIteratorResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);



inline const Aws::String& GetShardIterator() const{ return m_shardIterator; }


inline void SetShardIterator(const Aws::String& value) { m_shardIterator = value; }


inline void SetShardIterator(Aws::String&& value) { m_shardIterator = std::move(value); }


inline void SetShardIterator(const char* value) { m_shardIterator.assign(value); }


inline GetShardIteratorResult& WithShardIterator(const Aws::String& value) { SetShardIterator(value); return *this;}


inline GetShardIteratorResult& WithShardIterator(Aws::String&& value) { SetShardIterator(std::move(value)); return *this;}


inline GetShardIteratorResult& WithShardIterator(const char* value) { SetShardIterator(value); return *this;}

private:

Aws::String m_shardIterator;
};

} 
} 
} 
