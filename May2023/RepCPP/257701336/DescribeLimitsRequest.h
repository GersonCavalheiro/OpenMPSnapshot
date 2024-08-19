

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/kinesis/KinesisRequest.h>

namespace Aws
{
namespace Kinesis
{
namespace Model
{


class AWS_KINESIS_API DescribeLimitsRequest : public KinesisRequest
{
public:
DescribeLimitsRequest();

inline virtual const char* GetServiceRequestName() const override { return "DescribeLimits"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;

};

} 
} 
} 
