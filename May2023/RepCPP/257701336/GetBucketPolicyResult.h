

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/stream/ResponseStream.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <utility>

namespace Aws
{
template<typename RESULT_TYPE>
class AmazonWebServiceResult;

namespace S3
{
namespace Model
{
class AWS_S3_API GetBucketPolicyResult
{
public:
GetBucketPolicyResult();
GetBucketPolicyResult(GetBucketPolicyResult&&);
GetBucketPolicyResult& operator=(GetBucketPolicyResult&&);
GetBucketPolicyResult(const GetBucketPolicyResult&) = delete;
GetBucketPolicyResult& operator=(const GetBucketPolicyResult&) = delete;


GetBucketPolicyResult(Aws::AmazonWebServiceResult<Aws::Utils::Stream::ResponseStream>&& result);
GetBucketPolicyResult& operator=(Aws::AmazonWebServiceResult<Aws::Utils::Stream::ResponseStream>&& result);




inline Aws::IOStream& GetPolicy() { return m_policy.GetUnderlyingStream(); }


inline void ReplaceBody(Aws::IOStream* body) { m_policy = Aws::Utils::Stream::ResponseStream(body); }

private:

Aws::Utils::Stream::ResponseStream m_policy;
};

} 
} 
} 
