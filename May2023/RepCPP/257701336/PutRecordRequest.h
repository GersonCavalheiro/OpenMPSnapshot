

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/kinesis/KinesisRequest.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/Array.h>
#include <utility>

namespace Aws
{
namespace Kinesis
{
namespace Model
{


class AWS_KINESIS_API PutRecordRequest : public KinesisRequest
{
public:
PutRecordRequest();

inline virtual const char* GetServiceRequestName() const override { return "PutRecord"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetStreamName() const{ return m_streamName; }


inline void SetStreamName(const Aws::String& value) { m_streamNameHasBeenSet = true; m_streamName = value; }


inline void SetStreamName(Aws::String&& value) { m_streamNameHasBeenSet = true; m_streamName = std::move(value); }


inline void SetStreamName(const char* value) { m_streamNameHasBeenSet = true; m_streamName.assign(value); }


inline PutRecordRequest& WithStreamName(const Aws::String& value) { SetStreamName(value); return *this;}


inline PutRecordRequest& WithStreamName(Aws::String&& value) { SetStreamName(std::move(value)); return *this;}


inline PutRecordRequest& WithStreamName(const char* value) { SetStreamName(value); return *this;}



inline const Aws::Utils::ByteBuffer& GetData() const{ return m_data; }


inline void SetData(const Aws::Utils::ByteBuffer& value) { m_dataHasBeenSet = true; m_data = value; }


inline void SetData(Aws::Utils::ByteBuffer&& value) { m_dataHasBeenSet = true; m_data = std::move(value); }


inline PutRecordRequest& WithData(const Aws::Utils::ByteBuffer& value) { SetData(value); return *this;}


inline PutRecordRequest& WithData(Aws::Utils::ByteBuffer&& value) { SetData(std::move(value)); return *this;}



inline const Aws::String& GetPartitionKey() const{ return m_partitionKey; }


inline void SetPartitionKey(const Aws::String& value) { m_partitionKeyHasBeenSet = true; m_partitionKey = value; }


inline void SetPartitionKey(Aws::String&& value) { m_partitionKeyHasBeenSet = true; m_partitionKey = std::move(value); }


inline void SetPartitionKey(const char* value) { m_partitionKeyHasBeenSet = true; m_partitionKey.assign(value); }


inline PutRecordRequest& WithPartitionKey(const Aws::String& value) { SetPartitionKey(value); return *this;}


inline PutRecordRequest& WithPartitionKey(Aws::String&& value) { SetPartitionKey(std::move(value)); return *this;}


inline PutRecordRequest& WithPartitionKey(const char* value) { SetPartitionKey(value); return *this;}



inline const Aws::String& GetExplicitHashKey() const{ return m_explicitHashKey; }


inline void SetExplicitHashKey(const Aws::String& value) { m_explicitHashKeyHasBeenSet = true; m_explicitHashKey = value; }


inline void SetExplicitHashKey(Aws::String&& value) { m_explicitHashKeyHasBeenSet = true; m_explicitHashKey = std::move(value); }


inline void SetExplicitHashKey(const char* value) { m_explicitHashKeyHasBeenSet = true; m_explicitHashKey.assign(value); }


inline PutRecordRequest& WithExplicitHashKey(const Aws::String& value) { SetExplicitHashKey(value); return *this;}


inline PutRecordRequest& WithExplicitHashKey(Aws::String&& value) { SetExplicitHashKey(std::move(value)); return *this;}


inline PutRecordRequest& WithExplicitHashKey(const char* value) { SetExplicitHashKey(value); return *this;}



inline const Aws::String& GetSequenceNumberForOrdering() const{ return m_sequenceNumberForOrdering; }


inline void SetSequenceNumberForOrdering(const Aws::String& value) { m_sequenceNumberForOrderingHasBeenSet = true; m_sequenceNumberForOrdering = value; }


inline void SetSequenceNumberForOrdering(Aws::String&& value) { m_sequenceNumberForOrderingHasBeenSet = true; m_sequenceNumberForOrdering = std::move(value); }


inline void SetSequenceNumberForOrdering(const char* value) { m_sequenceNumberForOrderingHasBeenSet = true; m_sequenceNumberForOrdering.assign(value); }


inline PutRecordRequest& WithSequenceNumberForOrdering(const Aws::String& value) { SetSequenceNumberForOrdering(value); return *this;}


inline PutRecordRequest& WithSequenceNumberForOrdering(Aws::String&& value) { SetSequenceNumberForOrdering(std::move(value)); return *this;}


inline PutRecordRequest& WithSequenceNumberForOrdering(const char* value) { SetSequenceNumberForOrdering(value); return *this;}

private:

Aws::String m_streamName;
bool m_streamNameHasBeenSet;

Aws::Utils::ByteBuffer m_data;
bool m_dataHasBeenSet;

Aws::String m_partitionKey;
bool m_partitionKeyHasBeenSet;

Aws::String m_explicitHashKey;
bool m_explicitHashKeyHasBeenSet;

Aws::String m_sequenceNumberForOrdering;
bool m_sequenceNumberForOrderingHasBeenSet;
};

} 
} 
} 
