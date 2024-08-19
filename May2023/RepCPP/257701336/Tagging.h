

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/Tag.h>
#include <utility>

namespace Aws
{
namespace Utils
{
namespace Xml
{
class XmlNode;
} 
} 
namespace S3
{
namespace Model
{

class AWS_S3_API Tagging
{
public:
Tagging();
Tagging(const Aws::Utils::Xml::XmlNode& xmlNode);
Tagging& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::Vector<Tag>& GetTagSet() const{ return m_tagSet; }


inline void SetTagSet(const Aws::Vector<Tag>& value) { m_tagSetHasBeenSet = true; m_tagSet = value; }


inline void SetTagSet(Aws::Vector<Tag>&& value) { m_tagSetHasBeenSet = true; m_tagSet = std::move(value); }


inline Tagging& WithTagSet(const Aws::Vector<Tag>& value) { SetTagSet(value); return *this;}


inline Tagging& WithTagSet(Aws::Vector<Tag>&& value) { SetTagSet(std::move(value)); return *this;}


inline Tagging& AddTagSet(const Tag& value) { m_tagSetHasBeenSet = true; m_tagSet.push_back(value); return *this; }


inline Tagging& AddTagSet(Tag&& value) { m_tagSetHasBeenSet = true; m_tagSet.push_back(std::move(value)); return *this; }

private:

Aws::Vector<Tag> m_tagSet;
bool m_tagSetHasBeenSet;
};

} 
} 
} 
