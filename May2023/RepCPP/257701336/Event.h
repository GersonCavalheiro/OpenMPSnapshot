

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace S3
{
namespace Model
{
enum class Event
{
NOT_SET,
s3_ReducedRedundancyLostObject,
s3_ObjectCreated,
s3_ObjectCreated_Put,
s3_ObjectCreated_Post,
s3_ObjectCreated_Copy,
s3_ObjectCreated_CompleteMultipartUpload,
s3_ObjectRemoved,
s3_ObjectRemoved_Delete,
s3_ObjectRemoved_DeleteMarkerCreated
};

namespace EventMapper
{
AWS_S3_API Event GetEventForName(const Aws::String& name);

AWS_S3_API Aws::String GetNameForEvent(Event value);
} 
} 
} 
} 
