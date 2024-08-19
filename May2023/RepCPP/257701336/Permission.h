

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace S3
{
namespace Model
{
enum class Permission
{
NOT_SET,
FULL_CONTROL,
WRITE,
WRITE_ACP,
READ,
READ_ACP
};

namespace PermissionMapper
{
AWS_S3_API Permission GetPermissionForName(const Aws::String& name);

AWS_S3_API Aws::String GetNameForPermission(Permission value);
} 
} 
} 
} 
