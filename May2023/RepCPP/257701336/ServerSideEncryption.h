

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace S3
{
namespace Model
{
enum class ServerSideEncryption
{
NOT_SET,
AES256,
aws_kms
};

namespace ServerSideEncryptionMapper
{
AWS_S3_API ServerSideEncryption GetServerSideEncryptionForName(const Aws::String& name);

AWS_S3_API Aws::String GetNameForServerSideEncryption(ServerSideEncryption value);
} 
} 
} 
} 
