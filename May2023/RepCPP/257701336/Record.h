

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/DateTime.h>
#include <aws/core/utils/Array.h>
#include <aws/kinesis/model/EncryptionType.h>
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


class AWS_KINESIS_API Record
{
public:
Record();
Record(const Aws::Utils::Json::JsonValue& jsonValue);
Record& operator=(const Aws::Utils::Json::JsonValue& jsonValue);
Aws::Utils::Json::JsonValue Jsonize() const;



inline const Aws::String& GetSequenceNumber() const{ return m_sequenceNumber; }


inline void SetSequenceNumber(const Aws::String& value) { m_sequenceNumberHasBeenSet = true; m_sequenceNumber = value; }


inline void SetSequenceNumber(Aws::String&& value) { m_sequenceNumberHasBeenSet = true; m_sequenceNumber = std::move(value); }


inline void SetSequenceNumber(const char* value) { m_sequenceNumberHasBeenSet = true; m_sequenceNumber.assign(value); }


inline Record& WithSequenceNumber(const Aws::String& value) { SetSequenceNumber(value); return *this;}


inline Record& WithSequenceNumber(Aws::String&& value) { SetSequenceNumber(std::move(value)); return *this;}


inline Record& WithSequenceNumber(const char* value) { SetSequenceNumber(value); return *this;}



inline const Aws::Utils::DateTime& GetApproximateArrivalTimestamp() const{ return m_approximateArrivalTimestamp; }


inline void SetApproximateArrivalTimestamp(const Aws::Utils::DateTime& value) { m_approximateArrivalTimestampHasBeenSet = true; m_approximateArrivalTimestamp = value; }


inline void SetApproximateArrivalTimestamp(Aws::Utils::DateTime&& value) { m_approximateArrivalTimestampHasBeenSet = true; m_approximateArrivalTimestamp = std::move(value); }


inline Record& WithApproximateArrivalTimestamp(const Aws::Utils::DateTime& value) { SetApproximateArrivalTimestamp(value); return *this;}


inline Record& WithApproximateArrivalTimestamp(Aws::Utils::DateTime&& value) { SetApproximateArrivalTimestamp(std::move(value)); return *this;}



inline const Aws::Utils::ByteBuffer& GetData() const{ return m_data; }


inline void SetData(const Aws::Utils::ByteBuffer& value) { m_dataHasBeenSet = true; m_data = value; }


inline void SetData(Aws::Utils::ByteBuffer&& value) { m_dataHasBeenSet = true; m_data = std::move(value); }


inline Record& WithData(const Aws::Utils::ByteBuffer& value) { SetData(value); return *this;}


inline Record& WithData(Aws::Utils::ByteBuffer&& value) { SetData(std::move(value)); return *this;}



inline const Aws::String& GetPartitionKey() const{ return m_partitionKey; }


inline void SetPartitionKey(const Aws::String& value) { m_partitionKeyHasBeenSet = true; m_partitionKey = value; }


inline void SetPartitionKey(Aws::String&& value) { m_partitionKeyHasBeenSet = true; m_partitionKey = std::move(value); }


inline void SetPartitionKey(const char* value) { m_partitionKeyHasBeenSet = true; m_partitionKey.assign(value); }


inline Record& WithPartitionKey(const Aws::String& value) { SetPartitionKey(value); return *this;}


inline Record& WithPartitionKey(Aws::String&& value) { SetPartitionKey(std::move(value)); return *this;}


inline Record& WithPartitionKey(const char* value) { SetPartitionKey(value); return *this;}



inline const EncryptionType& GetEncryptionType() const{ return m_encryptionType; }


inline void SetEncryptionType(const EncryptionType& value) { m_encryptionTypeHasBeenSet = true; m_encryptionType = value; }


inline void SetEncryptionType(EncryptionType&& value) { m_encryptionTypeHasBeenSet = true; m_encryptionType = std::move(value); }


inline Record& WithEncryptionType(const EncryptionType& value) { SetEncryptionType(value); return *this;}


inline Record& WithEncryptionType(EncryptionType&& value) { SetEncryptionType(std::move(value)); return *this;}

private:

Aws::String m_sequenceNumber;
bool m_sequenceNumberHasBeenSet;

Aws::Utils::DateTime m_approximateArrivalTimestamp;
bool m_approximateArrivalTimestampHasBeenSet;

Aws::Utils::ByteBuffer m_data;
bool m_dataHasBeenSet;

Aws::String m_partitionKey;
bool m_partitionKeyHasBeenSet;

EncryptionType m_encryptionType;
bool m_encryptionTypeHasBeenSet;
};

} 
} 
} 
