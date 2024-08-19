

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/kinesis/KinesisRequest.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/kinesis/model/PutRecordsRequestEntry.h>
#include <utility>

namespace Aws
{
namespace Kinesis
{
namespace Model
{


class AWS_KINESIS_API PutRecordsRequest : public KinesisRequest
{
public:
PutRecordsRequest();

inline virtual const char* GetServiceRequestName() const override { return "PutRecords"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::Vector<PutRecordsRequestEntry>& GetRecords() const{ return m_records; }


inline void SetRecords(const Aws::Vector<PutRecordsRequestEntry>& value) { m_recordsHasBeenSet = true; m_records = value; }


inline void SetRecords(Aws::Vector<PutRecordsRequestEntry>&& value) { m_recordsHasBeenSet = true; m_records = std::move(value); }


inline PutRecordsRequest& WithRecords(const Aws::Vector<PutRecordsRequestEntry>& value) { SetRecords(value); return *this;}


inline PutRecordsRequest& WithRecords(Aws::Vector<PutRecordsRequestEntry>&& value) { SetRecords(std::move(value)); return *this;}


inline PutRecordsRequest& AddRecords(const PutRecordsRequestEntry& value) { m_recordsHasBeenSet = true; m_records.push_back(value); return *this; }


inline PutRecordsRequest& AddRecords(PutRecordsRequestEntry&& value) { m_recordsHasBeenSet = true; m_records.push_back(std::move(value)); return *this; }



inline const Aws::String& GetStreamName() const{ return m_streamName; }


inline void SetStreamName(const Aws::String& value) { m_streamNameHasBeenSet = true; m_streamName = value; }


inline void SetStreamName(Aws::String&& value) { m_streamNameHasBeenSet = true; m_streamName = std::move(value); }


inline void SetStreamName(const char* value) { m_streamNameHasBeenSet = true; m_streamName.assign(value); }


inline PutRecordsRequest& WithStreamName(const Aws::String& value) { SetStreamName(value); return *this;}


inline PutRecordsRequest& WithStreamName(Aws::String&& value) { SetStreamName(std::move(value)); return *this;}


inline PutRecordsRequest& WithStreamName(const char* value) { SetStreamName(value); return *this;}

private:

Aws::Vector<PutRecordsRequestEntry> m_records;
bool m_recordsHasBeenSet;

Aws::String m_streamName;
bool m_streamNameHasBeenSet;
};

} 
} 
} 
