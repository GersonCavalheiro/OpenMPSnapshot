

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/FilterRuleName.h>
#include <aws/core/utils/memory/stl/AWSString.h>
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


class AWS_S3_API FilterRule
{
public:
FilterRule();
FilterRule(const Aws::Utils::Xml::XmlNode& xmlNode);
FilterRule& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const FilterRuleName& GetName() const{ return m_name; }


inline void SetName(const FilterRuleName& value) { m_nameHasBeenSet = true; m_name = value; }


inline void SetName(FilterRuleName&& value) { m_nameHasBeenSet = true; m_name = std::move(value); }


inline FilterRule& WithName(const FilterRuleName& value) { SetName(value); return *this;}


inline FilterRule& WithName(FilterRuleName&& value) { SetName(std::move(value)); return *this;}



inline const Aws::String& GetValue() const{ return m_value; }


inline void SetValue(const Aws::String& value) { m_valueHasBeenSet = true; m_value = value; }


inline void SetValue(Aws::String&& value) { m_valueHasBeenSet = true; m_value = std::move(value); }


inline void SetValue(const char* value) { m_valueHasBeenSet = true; m_value.assign(value); }


inline FilterRule& WithValue(const Aws::String& value) { SetValue(value); return *this;}


inline FilterRule& WithValue(Aws::String&& value) { SetValue(std::move(value)); return *this;}


inline FilterRule& WithValue(const char* value) { SetValue(value); return *this;}

private:

FilterRuleName m_name;
bool m_nameHasBeenSet;

Aws::String m_value;
bool m_valueHasBeenSet;
};

} 
} 
} 
