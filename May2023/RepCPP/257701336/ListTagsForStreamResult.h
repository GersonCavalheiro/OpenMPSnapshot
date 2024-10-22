

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/kinesis/model/Tag.h>
#include <utility>

namespace Aws
{
template<typename RESULT_TYPE>
class AmazonWebServiceResult;

namespace Utils
{
namespace Json
{
class JsonValue;
} 
} 
namespace Kinesis
{
namespace Model
{

class AWS_KINESIS_API ListTagsForStreamResult
{
public:
ListTagsForStreamResult();
ListTagsForStreamResult(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);
ListTagsForStreamResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);



inline const Aws::Vector<Tag>& GetTags() const{ return m_tags; }


inline void SetTags(const Aws::Vector<Tag>& value) { m_tags = value; }


inline void SetTags(Aws::Vector<Tag>&& value) { m_tags = std::move(value); }


inline ListTagsForStreamResult& WithTags(const Aws::Vector<Tag>& value) { SetTags(value); return *this;}


inline ListTagsForStreamResult& WithTags(Aws::Vector<Tag>&& value) { SetTags(std::move(value)); return *this;}


inline ListTagsForStreamResult& AddTags(const Tag& value) { m_tags.push_back(value); return *this; }


inline ListTagsForStreamResult& AddTags(Tag&& value) { m_tags.push_back(std::move(value)); return *this; }



inline bool GetHasMoreTags() const{ return m_hasMoreTags; }


inline void SetHasMoreTags(bool value) { m_hasMoreTags = value; }


inline ListTagsForStreamResult& WithHasMoreTags(bool value) { SetHasMoreTags(value); return *this;}

private:

Aws::Vector<Tag> m_tags;

bool m_hasMoreTags;
};

} 
} 
} 
