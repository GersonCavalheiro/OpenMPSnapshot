

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace S3
{
namespace Model
{
enum class TransitionStorageClass
{
NOT_SET,
GLACIER,
STANDARD_IA
};

namespace TransitionStorageClassMapper
{
AWS_S3_API TransitionStorageClass GetTransitionStorageClassForName(const Aws::String& name);

AWS_S3_API Aws::String GetNameForTransitionStorageClass(TransitionStorageClass value);
} 
} 
} 
} 
