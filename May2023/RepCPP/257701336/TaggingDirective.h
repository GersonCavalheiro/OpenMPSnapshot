

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace S3
{
namespace Model
{
enum class TaggingDirective
{
NOT_SET,
COPY,
REPLACE
};

namespace TaggingDirectiveMapper
{
AWS_S3_API TaggingDirective GetTaggingDirectiveForName(const Aws::String& name);

AWS_S3_API Aws::String GetNameForTaggingDirective(TaggingDirective value);
} 
} 
} 
} 
