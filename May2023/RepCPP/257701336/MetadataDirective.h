

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace S3
{
namespace Model
{
enum class MetadataDirective
{
NOT_SET,
COPY,
REPLACE
};

namespace MetadataDirectiveMapper
{
AWS_S3_API MetadataDirective GetMetadataDirectiveForName(const Aws::String& name);

AWS_S3_API Aws::String GetNameForMetadataDirective(MetadataDirective value);
} 
} 
} 
} 
