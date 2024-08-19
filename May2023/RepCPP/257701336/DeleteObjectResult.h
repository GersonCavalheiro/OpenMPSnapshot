

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/RequestCharged.h>
#include <utility>

namespace Aws
{
template<typename RESULT_TYPE>
class AmazonWebServiceResult;

namespace Utils
{
namespace Xml
{
class XmlDocument;
} 
} 
namespace S3
{
namespace Model
{
class AWS_S3_API DeleteObjectResult
{
public:
DeleteObjectResult();
DeleteObjectResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
DeleteObjectResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline bool GetDeleteMarker() const{ return m_deleteMarker; }


inline void SetDeleteMarker(bool value) { m_deleteMarker = value; }


inline DeleteObjectResult& WithDeleteMarker(bool value) { SetDeleteMarker(value); return *this;}



inline const Aws::String& GetVersionId() const{ return m_versionId; }


inline void SetVersionId(const Aws::String& value) { m_versionId = value; }


inline void SetVersionId(Aws::String&& value) { m_versionId = std::move(value); }


inline void SetVersionId(const char* value) { m_versionId.assign(value); }


inline DeleteObjectResult& WithVersionId(const Aws::String& value) { SetVersionId(value); return *this;}


inline DeleteObjectResult& WithVersionId(Aws::String&& value) { SetVersionId(std::move(value)); return *this;}


inline DeleteObjectResult& WithVersionId(const char* value) { SetVersionId(value); return *this;}



inline const RequestCharged& GetRequestCharged() const{ return m_requestCharged; }


inline void SetRequestCharged(const RequestCharged& value) { m_requestCharged = value; }


inline void SetRequestCharged(RequestCharged&& value) { m_requestCharged = std::move(value); }


inline DeleteObjectResult& WithRequestCharged(const RequestCharged& value) { SetRequestCharged(value); return *this;}


inline DeleteObjectResult& WithRequestCharged(RequestCharged&& value) { SetRequestCharged(std::move(value)); return *this;}

private:

bool m_deleteMarker;

Aws::String m_versionId;

RequestCharged m_requestCharged;
};

} 
} 
} 
