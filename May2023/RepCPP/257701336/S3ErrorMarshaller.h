

#pragma once

#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/client/AWSErrorMarshaller.h>

namespace Aws
{
namespace Client
{

class AWS_S3_API S3ErrorMarshaller : public Aws::Client::XmlErrorMarshaller
{
public:
Aws::Client::AWSError<Aws::Client::CoreErrors> FindErrorByName(const char* exceptionName) const override;
};

} 
} 
