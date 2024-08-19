

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/Tag.h>
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
class AWS_S3_API GetBucketTaggingResult
{
public:
GetBucketTaggingResult();
GetBucketTaggingResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
GetBucketTaggingResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const Aws::Vector<Tag>& GetTagSet() const{ return m_tagSet; }


inline void SetTagSet(const Aws::Vector<Tag>& value) { m_tagSet = value; }


inline void SetTagSet(Aws::Vector<Tag>&& value) { m_tagSet = std::move(value); }


inline GetBucketTaggingResult& WithTagSet(const Aws::Vector<Tag>& value) { SetTagSet(value); return *this;}


inline GetBucketTaggingResult& WithTagSet(Aws::Vector<Tag>&& value) { SetTagSet(std::move(value)); return *this;}


inline GetBucketTaggingResult& AddTagSet(const Tag& value) { m_tagSet.push_back(value); return *this; }


inline GetBucketTaggingResult& AddTagSet(Tag&& value) { m_tagSet.push_back(std::move(value)); return *this; }

private:

Aws::Vector<Tag> m_tagSet;
};

} 
} 
} 
