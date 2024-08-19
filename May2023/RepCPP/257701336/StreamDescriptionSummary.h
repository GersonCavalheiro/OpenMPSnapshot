

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/kinesis/model/StreamStatus.h>
#include <aws/core/utils/DateTime.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/kinesis/model/EncryptionType.h>
#include <aws/kinesis/model/EnhancedMetrics.h>
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


class AWS_KINESIS_API StreamDescriptionSummary
{
public:
StreamDescriptionSummary();
StreamDescriptionSummary(const Aws::Utils::Json::JsonValue& jsonValue);
StreamDescriptionSummary& operator=(const Aws::Utils::Json::JsonValue& jsonValue);
Aws::Utils::Json::JsonValue Jsonize() const;



inline const Aws::String& GetStreamName() const{ return m_streamName; }


inline void SetStreamName(const Aws::String& value) { m_streamNameHasBeenSet = true; m_streamName = value; }


inline void SetStreamName(Aws::String&& value) { m_streamNameHasBeenSet = true; m_streamName = std::move(value); }


inline void SetStreamName(const char* value) { m_streamNameHasBeenSet = true; m_streamName.assign(value); }


inline StreamDescriptionSummary& WithStreamName(const Aws::String& value) { SetStreamName(value); return *this;}


inline StreamDescriptionSummary& WithStreamName(Aws::String&& value) { SetStreamName(std::move(value)); return *this;}


inline StreamDescriptionSummary& WithStreamName(const char* value) { SetStreamName(value); return *this;}



inline const Aws::String& GetStreamARN() const{ return m_streamARN; }


inline void SetStreamARN(const Aws::String& value) { m_streamARNHasBeenSet = true; m_streamARN = value; }


inline void SetStreamARN(Aws::String&& value) { m_streamARNHasBeenSet = true; m_streamARN = std::move(value); }


inline void SetStreamARN(const char* value) { m_streamARNHasBeenSet = true; m_streamARN.assign(value); }


inline StreamDescriptionSummary& WithStreamARN(const Aws::String& value) { SetStreamARN(value); return *this;}


inline StreamDescriptionSummary& WithStreamARN(Aws::String&& value) { SetStreamARN(std::move(value)); return *this;}


inline StreamDescriptionSummary& WithStreamARN(const char* value) { SetStreamARN(value); return *this;}



inline const StreamStatus& GetStreamStatus() const{ return m_streamStatus; }


inline void SetStreamStatus(const StreamStatus& value) { m_streamStatusHasBeenSet = true; m_streamStatus = value; }


inline void SetStreamStatus(StreamStatus&& value) { m_streamStatusHasBeenSet = true; m_streamStatus = std::move(value); }


inline StreamDescriptionSummary& WithStreamStatus(const StreamStatus& value) { SetStreamStatus(value); return *this;}


inline StreamDescriptionSummary& WithStreamStatus(StreamStatus&& value) { SetStreamStatus(std::move(value)); return *this;}



inline int GetRetentionPeriodHours() const{ return m_retentionPeriodHours; }


inline void SetRetentionPeriodHours(int value) { m_retentionPeriodHoursHasBeenSet = true; m_retentionPeriodHours = value; }


inline StreamDescriptionSummary& WithRetentionPeriodHours(int value) { SetRetentionPeriodHours(value); return *this;}



inline const Aws::Utils::DateTime& GetStreamCreationTimestamp() const{ return m_streamCreationTimestamp; }


inline void SetStreamCreationTimestamp(const Aws::Utils::DateTime& value) { m_streamCreationTimestampHasBeenSet = true; m_streamCreationTimestamp = value; }


inline void SetStreamCreationTimestamp(Aws::Utils::DateTime&& value) { m_streamCreationTimestampHasBeenSet = true; m_streamCreationTimestamp = std::move(value); }


inline StreamDescriptionSummary& WithStreamCreationTimestamp(const Aws::Utils::DateTime& value) { SetStreamCreationTimestamp(value); return *this;}


inline StreamDescriptionSummary& WithStreamCreationTimestamp(Aws::Utils::DateTime&& value) { SetStreamCreationTimestamp(std::move(value)); return *this;}



inline const Aws::Vector<EnhancedMetrics>& GetEnhancedMonitoring() const{ return m_enhancedMonitoring; }


inline void SetEnhancedMonitoring(const Aws::Vector<EnhancedMetrics>& value) { m_enhancedMonitoringHasBeenSet = true; m_enhancedMonitoring = value; }


inline void SetEnhancedMonitoring(Aws::Vector<EnhancedMetrics>&& value) { m_enhancedMonitoringHasBeenSet = true; m_enhancedMonitoring = std::move(value); }


inline StreamDescriptionSummary& WithEnhancedMonitoring(const Aws::Vector<EnhancedMetrics>& value) { SetEnhancedMonitoring(value); return *this;}


inline StreamDescriptionSummary& WithEnhancedMonitoring(Aws::Vector<EnhancedMetrics>&& value) { SetEnhancedMonitoring(std::move(value)); return *this;}


inline StreamDescriptionSummary& AddEnhancedMonitoring(const EnhancedMetrics& value) { m_enhancedMonitoringHasBeenSet = true; m_enhancedMonitoring.push_back(value); return *this; }


inline StreamDescriptionSummary& AddEnhancedMonitoring(EnhancedMetrics&& value) { m_enhancedMonitoringHasBeenSet = true; m_enhancedMonitoring.push_back(std::move(value)); return *this; }



inline const EncryptionType& GetEncryptionType() const{ return m_encryptionType; }


inline void SetEncryptionType(const EncryptionType& value) { m_encryptionTypeHasBeenSet = true; m_encryptionType = value; }


inline void SetEncryptionType(EncryptionType&& value) { m_encryptionTypeHasBeenSet = true; m_encryptionType = std::move(value); }


inline StreamDescriptionSummary& WithEncryptionType(const EncryptionType& value) { SetEncryptionType(value); return *this;}


inline StreamDescriptionSummary& WithEncryptionType(EncryptionType&& value) { SetEncryptionType(std::move(value)); return *this;}



inline const Aws::String& GetKeyId() const{ return m_keyId; }


inline void SetKeyId(const Aws::String& value) { m_keyIdHasBeenSet = true; m_keyId = value; }


inline void SetKeyId(Aws::String&& value) { m_keyIdHasBeenSet = true; m_keyId = std::move(value); }


inline void SetKeyId(const char* value) { m_keyIdHasBeenSet = true; m_keyId.assign(value); }


inline StreamDescriptionSummary& WithKeyId(const Aws::String& value) { SetKeyId(value); return *this;}


inline StreamDescriptionSummary& WithKeyId(Aws::String&& value) { SetKeyId(std::move(value)); return *this;}


inline StreamDescriptionSummary& WithKeyId(const char* value) { SetKeyId(value); return *this;}



inline int GetOpenShardCount() const{ return m_openShardCount; }


inline void SetOpenShardCount(int value) { m_openShardCountHasBeenSet = true; m_openShardCount = value; }


inline StreamDescriptionSummary& WithOpenShardCount(int value) { SetOpenShardCount(value); return *this;}

private:

Aws::String m_streamName;
bool m_streamNameHasBeenSet;

Aws::String m_streamARN;
bool m_streamARNHasBeenSet;

StreamStatus m_streamStatus;
bool m_streamStatusHasBeenSet;

int m_retentionPeriodHours;
bool m_retentionPeriodHoursHasBeenSet;

Aws::Utils::DateTime m_streamCreationTimestamp;
bool m_streamCreationTimestampHasBeenSet;

Aws::Vector<EnhancedMetrics> m_enhancedMonitoring;
bool m_enhancedMonitoringHasBeenSet;

EncryptionType m_encryptionType;
bool m_encryptionTypeHasBeenSet;

Aws::String m_keyId;
bool m_keyIdHasBeenSet;

int m_openShardCount;
bool m_openShardCountHasBeenSet;
};

} 
} 
} 
