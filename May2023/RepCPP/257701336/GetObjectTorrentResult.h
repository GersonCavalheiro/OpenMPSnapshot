

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/stream/ResponseStream.h>
#include <aws/core/utils/Array.h>
#include <aws/s3/model/RequestCharged.h>
#include <utility>

namespace Aws
{
template<typename RESULT_TYPE>
class AmazonWebServiceResult;

namespace S3
{
namespace Model
{
class AWS_S3_API GetObjectTorrentResult
{
public:
GetObjectTorrentResult();
GetObjectTorrentResult(GetObjectTorrentResult&&);
GetObjectTorrentResult& operator=(GetObjectTorrentResult&&);
GetObjectTorrentResult(const GetObjectTorrentResult&) = delete;
GetObjectTorrentResult& operator=(const GetObjectTorrentResult&) = delete;


GetObjectTorrentResult(Aws::AmazonWebServiceResult<Aws::Utils::Stream::ResponseStream>&& result);
GetObjectTorrentResult& operator=(Aws::AmazonWebServiceResult<Aws::Utils::Stream::ResponseStream>&& result);




inline Aws::IOStream& GetBody() { return m_body.GetUnderlyingStream(); }


inline void ReplaceBody(Aws::IOStream* body) { m_body = Aws::Utils::Stream::ResponseStream(body); }



inline const RequestCharged& GetRequestCharged() const{ return m_requestCharged; }


inline void SetRequestCharged(const RequestCharged& value) { m_requestCharged = value; }


inline void SetRequestCharged(RequestCharged&& value) { m_requestCharged = std::move(value); }


inline GetObjectTorrentResult& WithRequestCharged(const RequestCharged& value) { SetRequestCharged(value); return *this;}


inline GetObjectTorrentResult& WithRequestCharged(RequestCharged&& value) { SetRequestCharged(std::move(value)); return *this;}

private:

Aws::Utils::Stream::ResponseStream m_body;

RequestCharged m_requestCharged;
};

} 
} 
} 
