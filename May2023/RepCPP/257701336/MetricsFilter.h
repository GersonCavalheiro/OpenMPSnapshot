

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/Tag.h>
#include <aws/s3/model/MetricsAndOperator.h>
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

class AWS_S3_API MetricsFilter
{
public:
MetricsFilter();
MetricsFilter(const Aws::Utils::Xml::XmlNode& xmlNode);
MetricsFilter& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetPrefix() const{ return m_prefix; }


inline void SetPrefix(const Aws::String& value) { m_prefixHasBeenSet = true; m_prefix = value; }


inline void SetPrefix(Aws::String&& value) { m_prefixHasBeenSet = true; m_prefix = std::move(value); }


inline void SetPrefix(const char* value) { m_prefixHasBeenSet = true; m_prefix.assign(value); }


inline MetricsFilter& WithPrefix(const Aws::String& value) { SetPrefix(value); return *this;}


inline MetricsFilter& WithPrefix(Aws::String&& value) { SetPrefix(std::move(value)); return *this;}


inline MetricsFilter& WithPrefix(const char* value) { SetPrefix(value); return *this;}



inline const Tag& GetTag() const{ return m_tag; }


inline void SetTag(const Tag& value) { m_tagHasBeenSet = true; m_tag = value; }


inline void SetTag(Tag&& value) { m_tagHasBeenSet = true; m_tag = std::move(value); }


inline MetricsFilter& WithTag(const Tag& value) { SetTag(value); return *this;}


inline MetricsFilter& WithTag(Tag&& value) { SetTag(std::move(value)); return *this;}



inline const MetricsAndOperator& GetAnd() const{ return m_and; }


inline void SetAnd(const MetricsAndOperator& value) { m_andHasBeenSet = true; m_and = value; }


inline void SetAnd(MetricsAndOperator&& value) { m_andHasBeenSet = true; m_and = std::move(value); }


inline MetricsFilter& WithAnd(const MetricsAndOperator& value) { SetAnd(value); return *this;}


inline MetricsFilter& WithAnd(MetricsAndOperator&& value) { SetAnd(std::move(value)); return *this;}

private:

Aws::String m_prefix;
bool m_prefixHasBeenSet;

Tag m_tag;
bool m_tagHasBeenSet;

MetricsAndOperator m_and;
bool m_andHasBeenSet;
};

} 
} 
} 
