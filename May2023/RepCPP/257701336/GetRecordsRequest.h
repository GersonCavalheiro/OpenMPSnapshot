

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/kinesis/KinesisRequest.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <utility>

namespace Aws
{
namespace Kinesis
{
namespace Model
{


class AWS_KINESIS_API GetRecordsRequest : public KinesisRequest
{
public:
GetRecordsRequest();

inline virtual const char* GetServiceRequestName() const override { return "GetRecords"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetShardIterator() const{ return m_shardIterator; }


inline void SetShardIterator(const Aws::String& value) { m_shardIteratorHasBeenSet = true; m_shardIterator = value; }


inline void SetShardIterator(Aws::String&& value) { m_shardIteratorHasBeenSet = true; m_shardIterator = std::move(value); }


inline void SetShardIterator(const char* value) { m_shardIteratorHasBeenSet = true; m_shardIterator.assign(value); }


inline GetRecordsRequest& WithShardIterator(const Aws::String& value) { SetShardIterator(value); return *this;}


inline GetRecordsRequest& WithShardIterator(Aws::String&& value) { SetShardIterator(std::move(value)); return *this;}


inline GetRecordsRequest& WithShardIterator(const char* value) { SetShardIterator(value); return *this;}



inline int GetLimit() const{ return m_limit; }


inline void SetLimit(int value) { m_limitHasBeenSet = true; m_limit = value; }


inline GetRecordsRequest& WithLimit(int value) { SetLimit(value); return *this;}

private:

Aws::String m_shardIterator;
bool m_shardIteratorHasBeenSet;

int m_limit;
bool m_limitHasBeenSet;
};

} 
} 
} 
