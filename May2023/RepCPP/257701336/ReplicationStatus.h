

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace S3
{
namespace Model
{
enum class ReplicationStatus
{
NOT_SET,
COMPLETE,
PENDING,
FAILED,
REPLICA
};

namespace ReplicationStatusMapper
{
AWS_S3_API ReplicationStatus GetReplicationStatusForName(const Aws::String& name);

AWS_S3_API Aws::String GetNameForReplicationStatus(ReplicationStatus value);
} 
} 
} 
} 
