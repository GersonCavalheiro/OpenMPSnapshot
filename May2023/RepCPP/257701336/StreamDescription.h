

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/kinesis/model/StreamStatus.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/core/utils/DateTime.h>
#include <aws/kinesis/model/EncryptionType.h>
#include <aws/kinesis/model/Shard.h>
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


class AWS_KINESIS_API StreamDescription
{
public:
StreamDescription();
StreamDescription(const Aws::Utils::Json::JsonValue& jsonValue);
StreamDescription& operator=(const Aws::Utils::Json::JsonValue& jsonValue);
Aws::Utils::Json::JsonValue Jsonize() const;



inline const Aws::String& GetStreamName() const{ return m_streamName; }


inline void SetStreamName(const Aws::String& value) { m_streamNameHasBeenSet = true; m_streamName = value; }


inline void SetStreamName(Aws::String&& value) { m_streamNameHasBeenSet = true; m_streamName = std::move(value); }


inline void SetStreamName(const char* value) { m_streamNameHasBeenSet = true; m_streamName.assign(value); }


inline StreamDescription& WithStreamName(const Aws::String& value) { SetStreamName(value); return *this;}


inline StreamDescription& WithStreamName(Aws::String&& value) { SetStreamName(std::move(value)); return *this;}


inline StreamDescription& WithStreamName(const char* value) { SetStreamName(value); return *this;}



inline const Aws::String& GetStreamARN() const{ return m_streamARN; }


inline void SetStreamARN(const Aws::String& value) { m_streamARNHasBeenSet = true; m_streamARN = value; }


inline void SetStreamARN(Aws::String&& value) { m_streamARNHasBeenSet = true; m_streamARN = std::move(value); }


inline void SetStreamARN(const char* value) { m_streamARNHasBeenSet = true; m_streamARN.assign(value); }


inline StreamDescription& WithStreamARN(const Aws::String& value) { SetStreamARN(value); return *this;}


inline StreamDescription& WithStreamARN(Aws::String&& value) { SetStreamARN(std::move(value)); return *this;}


inline StreamDescription& WithStreamARN(const char* value) { SetStreamARN(value); return *this;}



inline const StreamStatus& GetStreamStatus() const{ return m_streamStatus; }


inline void SetStreamStatus(const StreamStatus& value) { m_streamStatusHasBeenSet = true; m_streamStatus = value; }


inline void SetStreamStatus(StreamStatus&& value) { m_streamStatusHasBeenSet = true; m_streamStatus = std::move(value); }


inline StreamDescription& WithStreamStatus(const StreamStatus& value) { SetStreamStatus(value); return *this;}


inline StreamDescription& WithStreamStatus(StreamStatus&& value) { SetStreamStatus(std::move(value)); return *this;}



inline const Aws::Vector<Shard>& GetShards() const{ return m_shards; }


inline void SetShards(const Aws::Vector<Shard>& value) { m_shardsHasBeenSet = true; m_shards = value; }


inline void SetShards(Aws::Vector<Shard>&& value) { m_shardsHasBeenSet = true; m_shards = std::move(value); }


inline StreamDescription& WithShards(const Aws::Vector<Shard>& value) { SetShards(value); return *this;}


inline StreamDescription& WithShards(Aws::Vector<Shard>&& value) { SetShards(std::move(value)); return *this;}


inline StreamDescription& AddShards(const Shard& value) { m_shardsHasBeenSet = true; m_shards.push_back(value); return *this; }


inline StreamDescription& AddShards(Shard&& value) { m_shardsHasBeenSet = true; m_shards.push_back(std::move(value)); return *this; }



inline bool GetHasMoreShards() const{ return m_hasMoreShards; }


inline void SetHasMoreShards(bool value) { m_hasMoreShardsHasBeenSet = true; m_hasMoreShards = value; }


inline StreamDescription& WithHasMoreShards(bool value) { SetHasMoreShards(value); return *this;}



inline int GetRetentionPeriodHours() const{ return m_retentionPeriodHours; }


inline void SetRetentionPeriodHours(int value) { m_retentionPeriodHoursHasBeenSet = true; m_retentionPeriodHours = value; }


inline StreamDescription& WithRetentionPeriodHours(int value) { SetRetentionPeriodHours(value); return *this;}



inline const Aws::Utils::DateTime& GetStreamCreationTimestamp() const{ return m_streamCreationTimestamp; }


inline void SetStreamCreationTimestamp(const Aws::Utils::DateTime& value) { m_streamCreationTimestampHasBeenSet = true; m_streamCreationTimestamp = value; }


inline void SetStreamCreationTimestamp(Aws::Utils::DateTime&& value) { m_streamCreationTimestampHasBeenSet = true; m_streamCreationTimestamp = std::move(value); }


inline StreamDescription& WithStreamCreationTimestamp(const Aws::Utils::DateTime& value) { SetStreamCreationTimestamp(value); return *this;}


inline StreamDescription& WithStreamCreationTimestamp(Aws::Utils::DateTime&& value) { SetStreamCreationTimestamp(std::move(value)); return *this;}



inline const Aws::Vector<EnhancedMetrics>& GetEnhancedMonitoring() const{ return m_enhancedMonitoring; }


inline void SetEnhancedMonitoring(const Aws::Vector<EnhancedMetrics>& value) { m_enhancedMonitoringHasBeenSet = true; m_enhancedMonitoring = value; }


inline void SetEnhancedMonitoring(Aws::Vector<EnhancedMetrics>&& value) { m_enhancedMonitoringHasBeenSet = true; m_enhancedMonitoring = std::move(value); }


inline StreamDescription& WithEnhancedMonitoring(const Aws::Vector<EnhancedMetrics>& value) { SetEnhancedMonitoring(value); return *this;}


inline StreamDescription& WithEnhancedMonitoring(Aws::Vector<EnhancedMetrics>&& value) { SetEnhancedMonitoring(std::move(value)); return *this;}


inline StreamDescription& AddEnhancedMonitoring(const EnhancedMetrics& value) { m_enhancedMonitoringHasBeenSet = true; m_enhancedMonitoring.push_back(value); return *this; }


inline StreamDescription& AddEnhancedMonitoring(EnhancedMetrics&& value) { m_enhancedMonitoringHasBeenSet = true; m_enhancedMonitoring.push_back(std::move(value)); return *this; }



inline const EncryptionType& GetEncryptionType() const{ return m_encryptionType; }


inline void SetEncryptionType(const EncryptionType& value) { m_encryptionTypeHasBeenSet = true; m_encryptionType = value; }


inline void SetEncryptionType(EncryptionType&& value) { m_encryptionTypeHasBeenSet = true; m_encryptionType = std::move(value); }


inline StreamDescription& WithEncryptionType(const EncryptionType& value) { SetEncryptionType(value); return *this;}


inline StreamDescription& WithEncryptionType(EncryptionType&& value) { SetEncryptionType(std::move(value)); return *this;}



inline const Aws::String& GetKeyId() const{ return m_keyId; }


inline void SetKeyId(const Aws::String& value) { m_keyIdHasBeenSet = true; m_keyId = value; }


inline void SetKeyId(Aws::String&& value) { m_keyIdHasBeenSet = true; m_keyId = std::move(value); }


inline void SetKeyId(const char* value) { m_keyIdHasBeenSet = true; m_keyId.assign(value); }


inline StreamDescription& WithKeyId(const Aws::String& value) { SetKeyId(value); return *this;}


inline StreamDescription& WithKeyId(Aws::String&& value) { SetKeyId(std::move(value)); return *this;}


inline StreamDescription& WithKeyId(const char* value) { SetKeyId(value); return *this;}

private:

Aws::String m_streamName;
bool m_streamNameHasBeenSet;

Aws::String m_streamARN;
bool m_streamARNHasBeenSet;

StreamStatus m_streamStatus;
bool m_streamStatusHasBeenSet;

Aws::Vector<Shard> m_shards;
bool m_shardsHasBeenSet;

bool m_hasMoreShards;
bool m_hasMoreShardsHasBeenSet;

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
};

} 
} 
} 
