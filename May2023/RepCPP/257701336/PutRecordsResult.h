

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/kinesis/model/EncryptionType.h>
#include <aws/kinesis/model/PutRecordsResultEntry.h>
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

class AWS_KINESIS_API PutRecordsResult
{
public:
PutRecordsResult();
PutRecordsResult(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);
PutRecordsResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);



inline int GetFailedRecordCount() const{ return m_failedRecordCount; }


inline void SetFailedRecordCount(int value) { m_failedRecordCount = value; }


inline PutRecordsResult& WithFailedRecordCount(int value) { SetFailedRecordCount(value); return *this;}



inline const Aws::Vector<PutRecordsResultEntry>& GetRecords() const{ return m_records; }


inline void SetRecords(const Aws::Vector<PutRecordsResultEntry>& value) { m_records = value; }


inline void SetRecords(Aws::Vector<PutRecordsResultEntry>&& value) { m_records = std::move(value); }


inline PutRecordsResult& WithRecords(const Aws::Vector<PutRecordsResultEntry>& value) { SetRecords(value); return *this;}


inline PutRecordsResult& WithRecords(Aws::Vector<PutRecordsResultEntry>&& value) { SetRecords(std::move(value)); return *this;}


inline PutRecordsResult& AddRecords(const PutRecordsResultEntry& value) { m_records.push_back(value); return *this; }


inline PutRecordsResult& AddRecords(PutRecordsResultEntry&& value) { m_records.push_back(std::move(value)); return *this; }



inline const EncryptionType& GetEncryptionType() const{ return m_encryptionType; }


inline void SetEncryptionType(const EncryptionType& value) { m_encryptionType = value; }


inline void SetEncryptionType(EncryptionType&& value) { m_encryptionType = std::move(value); }


inline PutRecordsResult& WithEncryptionType(const EncryptionType& value) { SetEncryptionType(value); return *this;}


inline PutRecordsResult& WithEncryptionType(EncryptionType&& value) { SetEncryptionType(std::move(value)); return *this;}

private:

int m_failedRecordCount;

Aws::Vector<PutRecordsResultEntry> m_records;

EncryptionType m_encryptionType;
};

} 
} 
} 
