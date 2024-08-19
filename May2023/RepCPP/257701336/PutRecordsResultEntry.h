

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
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


class AWS_KINESIS_API PutRecordsResultEntry
{
public:
PutRecordsResultEntry();
PutRecordsResultEntry(const Aws::Utils::Json::JsonValue& jsonValue);
PutRecordsResultEntry& operator=(const Aws::Utils::Json::JsonValue& jsonValue);
Aws::Utils::Json::JsonValue Jsonize() const;



inline const Aws::String& GetSequenceNumber() const{ return m_sequenceNumber; }


inline void SetSequenceNumber(const Aws::String& value) { m_sequenceNumberHasBeenSet = true; m_sequenceNumber = value; }


inline void SetSequenceNumber(Aws::String&& value) { m_sequenceNumberHasBeenSet = true; m_sequenceNumber = std::move(value); }


inline void SetSequenceNumber(const char* value) { m_sequenceNumberHasBeenSet = true; m_sequenceNumber.assign(value); }


inline PutRecordsResultEntry& WithSequenceNumber(const Aws::String& value) { SetSequenceNumber(value); return *this;}


inline PutRecordsResultEntry& WithSequenceNumber(Aws::String&& value) { SetSequenceNumber(std::move(value)); return *this;}


inline PutRecordsResultEntry& WithSequenceNumber(const char* value) { SetSequenceNumber(value); return *this;}



inline const Aws::String& GetShardId() const{ return m_shardId; }


inline void SetShardId(const Aws::String& value) { m_shardIdHasBeenSet = true; m_shardId = value; }


inline void SetShardId(Aws::String&& value) { m_shardIdHasBeenSet = true; m_shardId = std::move(value); }


inline void SetShardId(const char* value) { m_shardIdHasBeenSet = true; m_shardId.assign(value); }


inline PutRecordsResultEntry& WithShardId(const Aws::String& value) { SetShardId(value); return *this;}


inline PutRecordsResultEntry& WithShardId(Aws::String&& value) { SetShardId(std::move(value)); return *this;}


inline PutRecordsResultEntry& WithShardId(const char* value) { SetShardId(value); return *this;}



inline const Aws::String& GetErrorCode() const{ return m_errorCode; }


inline void SetErrorCode(const Aws::String& value) { m_errorCodeHasBeenSet = true; m_errorCode = value; }


inline void SetErrorCode(Aws::String&& value) { m_errorCodeHasBeenSet = true; m_errorCode = std::move(value); }


inline void SetErrorCode(const char* value) { m_errorCodeHasBeenSet = true; m_errorCode.assign(value); }


inline PutRecordsResultEntry& WithErrorCode(const Aws::String& value) { SetErrorCode(value); return *this;}


inline PutRecordsResultEntry& WithErrorCode(Aws::String&& value) { SetErrorCode(std::move(value)); return *this;}


inline PutRecordsResultEntry& WithErrorCode(const char* value) { SetErrorCode(value); return *this;}



inline const Aws::String& GetErrorMessage() const{ return m_errorMessage; }


inline void SetErrorMessage(const Aws::String& value) { m_errorMessageHasBeenSet = true; m_errorMessage = value; }


inline void SetErrorMessage(Aws::String&& value) { m_errorMessageHasBeenSet = true; m_errorMessage = std::move(value); }


inline void SetErrorMessage(const char* value) { m_errorMessageHasBeenSet = true; m_errorMessage.assign(value); }


inline PutRecordsResultEntry& WithErrorMessage(const Aws::String& value) { SetErrorMessage(value); return *this;}


inline PutRecordsResultEntry& WithErrorMessage(Aws::String&& value) { SetErrorMessage(std::move(value)); return *this;}


inline PutRecordsResultEntry& WithErrorMessage(const char* value) { SetErrorMessage(value); return *this;}

private:

Aws::String m_sequenceNumber;
bool m_sequenceNumberHasBeenSet;

Aws::String m_shardId;
bool m_shardIdHasBeenSet;

Aws::String m_errorCode;
bool m_errorCodeHasBeenSet;

Aws::String m_errorMessage;
bool m_errorMessageHasBeenSet;
};

} 
} 
} 
