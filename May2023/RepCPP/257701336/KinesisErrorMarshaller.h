

#pragma once

#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/client/AWSErrorMarshaller.h>

namespace Aws
{
namespace Client
{

class AWS_KINESIS_API KinesisErrorMarshaller : public Aws::Client::JsonErrorMarshaller
{
public:
Aws::Client::AWSError<Aws::Client::CoreErrors> FindErrorByName(const char* exceptionName) const override;
};

} 
} 
