

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
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
class AWS_S3_API DeleteObjectTaggingResult
{
public:
DeleteObjectTaggingResult();
DeleteObjectTaggingResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
DeleteObjectTaggingResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const Aws::String& GetVersionId() const{ return m_versionId; }


inline void SetVersionId(const Aws::String& value) { m_versionId = value; }


inline void SetVersionId(Aws::String&& value) { m_versionId = std::move(value); }


inline void SetVersionId(const char* value) { m_versionId.assign(value); }


inline DeleteObjectTaggingResult& WithVersionId(const Aws::String& value) { SetVersionId(value); return *this;}


inline DeleteObjectTaggingResult& WithVersionId(Aws::String&& value) { SetVersionId(std::move(value)); return *this;}


inline DeleteObjectTaggingResult& WithVersionId(const char* value) { SetVersionId(value); return *this;}

private:

Aws::String m_versionId;
};

} 
} 
} 
