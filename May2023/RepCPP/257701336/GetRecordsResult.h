

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/kinesis/model/Record.h>
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

class AWS_KINESIS_API GetRecordsResult
{
public:
GetRecordsResult();
GetRecordsResult(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);
GetRecordsResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);



inline const Aws::Vector<Record>& GetRecords() const{ return m_records; }


inline void SetRecords(const Aws::Vector<Record>& value) { m_records = value; }


inline void SetRecords(Aws::Vector<Record>&& value) { m_records = std::move(value); }


inline GetRecordsResult& WithRecords(const Aws::Vector<Record>& value) { SetRecords(value); return *this;}


inline GetRecordsResult& WithRecords(Aws::Vector<Record>&& value) { SetRecords(std::move(value)); return *this;}


inline GetRecordsResult& AddRecords(const Record& value) { m_records.push_back(value); return *this; }


inline GetRecordsResult& AddRecords(Record&& value) { m_records.push_back(std::move(value)); return *this; }



inline const Aws::String& GetNextShardIterator() const{ return m_nextShardIterator; }


inline void SetNextShardIterator(const Aws::String& value) { m_nextShardIterator = value; }


inline void SetNextShardIterator(Aws::String&& value) { m_nextShardIterator = std::move(value); }


inline void SetNextShardIterator(const char* value) { m_nextShardIterator.assign(value); }


inline GetRecordsResult& WithNextShardIterator(const Aws::String& value) { SetNextShardIterator(value); return *this;}


inline GetRecordsResult& WithNextShardIterator(Aws::String&& value) { SetNextShardIterator(std::move(value)); return *this;}


inline GetRecordsResult& WithNextShardIterator(const char* value) { SetNextShardIterator(value); return *this;}



inline long long GetMillisBehindLatest() const{ return m_millisBehindLatest; }


inline void SetMillisBehindLatest(long long value) { m_millisBehindLatest = value; }


inline GetRecordsResult& WithMillisBehindLatest(long long value) { SetMillisBehindLatest(value); return *this;}

private:

Aws::Vector<Record> m_records;

Aws::String m_nextShardIterator;

long long m_millisBehindLatest;
};

} 
} 
} 
