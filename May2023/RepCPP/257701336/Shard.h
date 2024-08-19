

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/kinesis/model/HashKeyRange.h>
#include <aws/kinesis/model/SequenceNumberRange.h>
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


class AWS_KINESIS_API Shard
{
public:
Shard();
Shard(const Aws::Utils::Json::JsonValue& jsonValue);
Shard& operator=(const Aws::Utils::Json::JsonValue& jsonValue);
Aws::Utils::Json::JsonValue Jsonize() const;



inline const Aws::String& GetShardId() const{ return m_shardId; }


inline void SetShardId(const Aws::String& value) { m_shardIdHasBeenSet = true; m_shardId = value; }


inline void SetShardId(Aws::String&& value) { m_shardIdHasBeenSet = true; m_shardId = std::move(value); }


inline void SetShardId(const char* value) { m_shardIdHasBeenSet = true; m_shardId.assign(value); }


inline Shard& WithShardId(const Aws::String& value) { SetShardId(value); return *this;}


inline Shard& WithShardId(Aws::String&& value) { SetShardId(std::move(value)); return *this;}


inline Shard& WithShardId(const char* value) { SetShardId(value); return *this;}



inline const Aws::String& GetParentShardId() const{ return m_parentShardId; }


inline void SetParentShardId(const Aws::String& value) { m_parentShardIdHasBeenSet = true; m_parentShardId = value; }


inline void SetParentShardId(Aws::String&& value) { m_parentShardIdHasBeenSet = true; m_parentShardId = std::move(value); }


inline void SetParentShardId(const char* value) { m_parentShardIdHasBeenSet = true; m_parentShardId.assign(value); }


inline Shard& WithParentShardId(const Aws::String& value) { SetParentShardId(value); return *this;}


inline Shard& WithParentShardId(Aws::String&& value) { SetParentShardId(std::move(value)); return *this;}


inline Shard& WithParentShardId(const char* value) { SetParentShardId(value); return *this;}



inline const Aws::String& GetAdjacentParentShardId() const{ return m_adjacentParentShardId; }


inline void SetAdjacentParentShardId(const Aws::String& value) { m_adjacentParentShardIdHasBeenSet = true; m_adjacentParentShardId = value; }


inline void SetAdjacentParentShardId(Aws::String&& value) { m_adjacentParentShardIdHasBeenSet = true; m_adjacentParentShardId = std::move(value); }


inline void SetAdjacentParentShardId(const char* value) { m_adjacentParentShardIdHasBeenSet = true; m_adjacentParentShardId.assign(value); }


inline Shard& WithAdjacentParentShardId(const Aws::String& value) { SetAdjacentParentShardId(value); return *this;}


inline Shard& WithAdjacentParentShardId(Aws::String&& value) { SetAdjacentParentShardId(std::move(value)); return *this;}


inline Shard& WithAdjacentParentShardId(const char* value) { SetAdjacentParentShardId(value); return *this;}



inline const HashKeyRange& GetHashKeyRange() const{ return m_hashKeyRange; }


inline void SetHashKeyRange(const HashKeyRange& value) { m_hashKeyRangeHasBeenSet = true; m_hashKeyRange = value; }


inline void SetHashKeyRange(HashKeyRange&& value) { m_hashKeyRangeHasBeenSet = true; m_hashKeyRange = std::move(value); }


inline Shard& WithHashKeyRange(const HashKeyRange& value) { SetHashKeyRange(value); return *this;}


inline Shard& WithHashKeyRange(HashKeyRange&& value) { SetHashKeyRange(std::move(value)); return *this;}



inline const SequenceNumberRange& GetSequenceNumberRange() const{ return m_sequenceNumberRange; }


inline void SetSequenceNumberRange(const SequenceNumberRange& value) { m_sequenceNumberRangeHasBeenSet = true; m_sequenceNumberRange = value; }


inline void SetSequenceNumberRange(SequenceNumberRange&& value) { m_sequenceNumberRangeHasBeenSet = true; m_sequenceNumberRange = std::move(value); }


inline Shard& WithSequenceNumberRange(const SequenceNumberRange& value) { SetSequenceNumberRange(value); return *this;}


inline Shard& WithSequenceNumberRange(SequenceNumberRange&& value) { SetSequenceNumberRange(std::move(value)); return *this;}

private:

Aws::String m_shardId;
bool m_shardIdHasBeenSet;

Aws::String m_parentShardId;
bool m_parentShardIdHasBeenSet;

Aws::String m_adjacentParentShardId;
bool m_adjacentParentShardIdHasBeenSet;

HashKeyRange m_hashKeyRange;
bool m_hashKeyRangeHasBeenSet;

SequenceNumberRange m_sequenceNumberRange;
bool m_sequenceNumberRangeHasBeenSet;
};

} 
} 
} 
