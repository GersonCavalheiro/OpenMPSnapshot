

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
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

class AWS_S3_API MetricsAndOperator
{
public:
MetricsAndOperator();
MetricsAndOperator(const Aws::Utils::Xml::XmlNode& xmlNode);
MetricsAndOperator& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetPrefix() const{ return m_prefix; }


inline void SetPrefix(const Aws::String& value) { m_prefixHasBeenSet = true; m_prefix = value; }


inline void SetPrefix(Aws::String&& value) { m_prefixHasBeenSet = true; m_prefix = std::move(value); }


inline void SetPrefix(const char* value) { m_prefixHasBeenSet = true; m_prefix.assign(value); }


inline MetricsAndOperator& WithPrefix(const Aws::String& value) { SetPrefix(value); return *this;}


inline MetricsAndOperator& WithPrefix(Aws::String&& value) { SetPrefix(std::move(value)); return *this;}


inline MetricsAndOperator& WithPrefix(const char* value) { SetPrefix(value); return *this;}



inline const Aws::Vector<Tag>& GetTags() const{ return m_tags; }


inline void SetTags(const Aws::Vector<Tag>& value) { m_tagsHasBeenSet = true; m_tags = value; }


inline void SetTags(Aws::Vector<Tag>&& value) { m_tagsHasBeenSet = true; m_tags = std::move(value); }


inline MetricsAndOperator& WithTags(const Aws::Vector<Tag>& value) { SetTags(value); return *this;}


inline MetricsAndOperator& WithTags(Aws::Vector<Tag>&& value) { SetTags(std::move(value)); return *this;}


inline MetricsAndOperator& AddTags(const Tag& value) { m_tagsHasBeenSet = true; m_tags.push_back(value); return *this; }


inline MetricsAndOperator& AddTags(Tag&& value) { m_tagsHasBeenSet = true; m_tags.push_back(std::move(value)); return *this; }

private:

Aws::String m_prefix;
bool m_prefixHasBeenSet;

Aws::Vector<Tag> m_tags;
bool m_tagsHasBeenSet;
};

} 
} 
} 
