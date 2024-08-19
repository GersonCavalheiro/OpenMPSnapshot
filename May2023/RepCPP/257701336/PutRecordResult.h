

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/kinesis/model/EncryptionType.h>
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

class AWS_KINESIS_API PutRecordResult
{
public:
PutRecordResult();
PutRecordResult(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);
PutRecordResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);



inline const Aws::String& GetShardId() const{ return m_shardId; }


inline void SetShardId(const Aws::String& value) { m_shardId = value; }


inline void SetShardId(Aws::String&& value) { m_shardId = std::move(value); }


inline void SetShardId(const char* value) { m_shardId.assign(value); }


inline PutRecordResult& WithShardId(const Aws::String& value) { SetShardId(value); return *this;}


inline PutRecordResult& WithShardId(Aws::String&& value) { SetShardId(std::move(value)); return *this;}


inline PutRecordResult& WithShardId(const char* value) { SetShardId(value); return *this;}



inline const Aws::String& GetSequenceNumber() const{ return m_sequenceNumber; }


inline void SetSequenceNumber(const Aws::String& value) { m_sequenceNumber = value; }


inline void SetSequenceNumber(Aws::String&& value) { m_sequenceNumber = std::move(value); }


inline void SetSequenceNumber(const char* value) { m_sequenceNumber.assign(value); }


inline PutRecordResult& WithSequenceNumber(const Aws::String& value) { SetSequenceNumber(value); return *this;}


inline PutRecordResult& WithSequenceNumber(Aws::String&& value) { SetSequenceNumber(std::move(value)); return *this;}


inline PutRecordResult& WithSequenceNumber(const char* value) { SetSequenceNumber(value); return *this;}



inline const EncryptionType& GetEncryptionType() const{ return m_encryptionType; }


inline void SetEncryptionType(const EncryptionType& value) { m_encryptionType = value; }


inline void SetEncryptionType(EncryptionType&& value) { m_encryptionType = std::move(value); }


inline PutRecordResult& WithEncryptionType(const EncryptionType& value) { SetEncryptionType(value); return *this;}


inline PutRecordResult& WithEncryptionType(EncryptionType&& value) { SetEncryptionType(std::move(value)); return *this;}

private:

Aws::String m_shardId;

Aws::String m_sequenceNumber;

EncryptionType m_encryptionType;
};

} 
} 
} 
