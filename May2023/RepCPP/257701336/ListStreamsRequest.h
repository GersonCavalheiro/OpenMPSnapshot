

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


class AWS_KINESIS_API ListStreamsRequest : public KinesisRequest
{
public:
ListStreamsRequest();

inline virtual const char* GetServiceRequestName() const override { return "ListStreams"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline int GetLimit() const{ return m_limit; }


inline void SetLimit(int value) { m_limitHasBeenSet = true; m_limit = value; }


inline ListStreamsRequest& WithLimit(int value) { SetLimit(value); return *this;}



inline const Aws::String& GetExclusiveStartStreamName() const{ return m_exclusiveStartStreamName; }


inline void SetExclusiveStartStreamName(const Aws::String& value) { m_exclusiveStartStreamNameHasBeenSet = true; m_exclusiveStartStreamName = value; }


inline void SetExclusiveStartStreamName(Aws::String&& value) { m_exclusiveStartStreamNameHasBeenSet = true; m_exclusiveStartStreamName = std::move(value); }


inline void SetExclusiveStartStreamName(const char* value) { m_exclusiveStartStreamNameHasBeenSet = true; m_exclusiveStartStreamName.assign(value); }


inline ListStreamsRequest& WithExclusiveStartStreamName(const Aws::String& value) { SetExclusiveStartStreamName(value); return *this;}


inline ListStreamsRequest& WithExclusiveStartStreamName(Aws::String&& value) { SetExclusiveStartStreamName(std::move(value)); return *this;}


inline ListStreamsRequest& WithExclusiveStartStreamName(const char* value) { SetExclusiveStartStreamName(value); return *this;}

private:

int m_limit;
bool m_limitHasBeenSet;

Aws::String m_exclusiveStartStreamName;
bool m_exclusiveStartStreamNameHasBeenSet;
};

} 
} 
} 
