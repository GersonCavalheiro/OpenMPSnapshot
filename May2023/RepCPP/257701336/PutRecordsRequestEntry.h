

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/utils/Array.h>
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


class AWS_KINESIS_API PutRecordsRequestEntry
{
public:
PutRecordsRequestEntry();
PutRecordsRequestEntry(const Aws::Utils::Json::JsonValue& jsonValue);
PutRecordsRequestEntry& operator=(const Aws::Utils::Json::JsonValue& jsonValue);
Aws::Utils::Json::JsonValue Jsonize() const;



inline const Aws::Utils::ByteBuffer& GetData() const{ return m_data; }


inline void SetData(const Aws::Utils::ByteBuffer& value) { m_dataHasBeenSet = true; m_data = value; }


inline void SetData(Aws::Utils::ByteBuffer&& value) { m_dataHasBeenSet = true; m_data = std::move(value); }


inline PutRecordsRequestEntry& WithData(const Aws::Utils::ByteBuffer& value) { SetData(value); return *this;}


inline PutRecordsRequestEntry& WithData(Aws::Utils::ByteBuffer&& value) { SetData(std::move(value)); return *this;}



inline const Aws::String& GetExplicitHashKey() const{ return m_explicitHashKey; }


inline void SetExplicitHashKey(const Aws::String& value) { m_explicitHashKeyHasBeenSet = true; m_explicitHashKey = value; }


inline void SetExplicitHashKey(Aws::String&& value) { m_explicitHashKeyHasBeenSet = true; m_explicitHashKey = std::move(value); }


inline void SetExplicitHashKey(const char* value) { m_explicitHashKeyHasBeenSet = true; m_explicitHashKey.assign(value); }


inline PutRecordsRequestEntry& WithExplicitHashKey(const Aws::String& value) { SetExplicitHashKey(value); return *this;}


inline PutRecordsRequestEntry& WithExplicitHashKey(Aws::String&& value) { SetExplicitHashKey(std::move(value)); return *this;}


inline PutRecordsRequestEntry& WithExplicitHashKey(const char* value) { SetExplicitHashKey(value); return *this;}



inline const Aws::String& GetPartitionKey() const{ return m_partitionKey; }


inline void SetPartitionKey(const Aws::String& value) { m_partitionKeyHasBeenSet = true; m_partitionKey = value; }


inline void SetPartitionKey(Aws::String&& value) { m_partitionKeyHasBeenSet = true; m_partitionKey = std::move(value); }


inline void SetPartitionKey(const char* value) { m_partitionKeyHasBeenSet = true; m_partitionKey.assign(value); }


inline PutRecordsRequestEntry& WithPartitionKey(const Aws::String& value) { SetPartitionKey(value); return *this;}


inline PutRecordsRequestEntry& WithPartitionKey(Aws::String&& value) { SetPartitionKey(std::move(value)); return *this;}


inline PutRecordsRequestEntry& WithPartitionKey(const char* value) { SetPartitionKey(value); return *this;}

private:

Aws::Utils::ByteBuffer m_data;
bool m_dataHasBeenSet;

Aws::String m_explicitHashKey;
bool m_explicitHashKeyHasBeenSet;

Aws::String m_partitionKey;
bool m_partitionKeyHasBeenSet;
};

} 
} 
} 
