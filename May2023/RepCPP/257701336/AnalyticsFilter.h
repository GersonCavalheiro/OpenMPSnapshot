

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/Tag.h>
#include <aws/s3/model/AnalyticsAndOperator.h>
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

class AWS_S3_API AnalyticsFilter
{
public:
AnalyticsFilter();
AnalyticsFilter(const Aws::Utils::Xml::XmlNode& xmlNode);
AnalyticsFilter& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetPrefix() const{ return m_prefix; }


inline void SetPrefix(const Aws::String& value) { m_prefixHasBeenSet = true; m_prefix = value; }


inline void SetPrefix(Aws::String&& value) { m_prefixHasBeenSet = true; m_prefix = std::move(value); }


inline void SetPrefix(const char* value) { m_prefixHasBeenSet = true; m_prefix.assign(value); }


inline AnalyticsFilter& WithPrefix(const Aws::String& value) { SetPrefix(value); return *this;}


inline AnalyticsFilter& WithPrefix(Aws::String&& value) { SetPrefix(std::move(value)); return *this;}


inline AnalyticsFilter& WithPrefix(const char* value) { SetPrefix(value); return *this;}



inline const Tag& GetTag() const{ return m_tag; }


inline void SetTag(const Tag& value) { m_tagHasBeenSet = true; m_tag = value; }


inline void SetTag(Tag&& value) { m_tagHasBeenSet = true; m_tag = std::move(value); }


inline AnalyticsFilter& WithTag(const Tag& value) { SetTag(value); return *this;}


inline AnalyticsFilter& WithTag(Tag&& value) { SetTag(std::move(value)); return *this;}



inline const AnalyticsAndOperator& GetAnd() const{ return m_and; }


inline void SetAnd(const AnalyticsAndOperator& value) { m_andHasBeenSet = true; m_and = value; }


inline void SetAnd(AnalyticsAndOperator&& value) { m_andHasBeenSet = true; m_and = std::move(value); }


inline AnalyticsFilter& WithAnd(const AnalyticsAndOperator& value) { SetAnd(value); return *this;}


inline AnalyticsFilter& WithAnd(AnalyticsAndOperator&& value) { SetAnd(std::move(value)); return *this;}

private:

Aws::String m_prefix;
bool m_prefixHasBeenSet;

Tag m_tag;
bool m_tagHasBeenSet;

AnalyticsAndOperator m_and;
bool m_andHasBeenSet;
};

} 
} 
} 
