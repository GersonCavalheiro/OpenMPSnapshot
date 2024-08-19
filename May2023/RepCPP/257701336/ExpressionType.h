

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace S3
{
namespace Model
{
enum class ExpressionType
{
NOT_SET,
SQL
};

namespace ExpressionTypeMapper
{
AWS_S3_API ExpressionType GetExpressionTypeForName(const Aws::String& name);

AWS_S3_API Aws::String GetNameForExpressionType(ExpressionType value);
} 
} 
} 
} 
