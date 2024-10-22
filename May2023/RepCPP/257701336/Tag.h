

#pragma once
#include <aws/s3/S3_EXPORTS.h>
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

class AWS_S3_API Tag
{
public:
Tag();
Tag(const Aws::Utils::Xml::XmlNode& xmlNode);
Tag& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetKey() const{ return m_key; }


inline void SetKey(const Aws::String& value) { m_keyHasBeenSet = true; m_key = value; }


inline void SetKey(Aws::String&& value) { m_keyHasBeenSet = true; m_key = std::move(value); }


inline void SetKey(const char* value) { m_keyHasBeenSet = true; m_key.assign(value); }


inline Tag& WithKey(const Aws::String& value) { SetKey(value); return *this;}


inline Tag& WithKey(Aws::String&& value) { SetKey(std::move(value)); return *this;}


inline Tag& WithKey(const char* value) { SetKey(value); return *this;}



inline const Aws::String& GetValue() const{ return m_value; }


inline void SetValue(const Aws::String& value) { m_valueHasBeenSet = true; m_value = value; }


inline void SetValue(Aws::String&& value) { m_valueHasBeenSet = true; m_value = std::move(value); }


inline void SetValue(const char* value) { m_valueHasBeenSet = true; m_value.assign(value); }


inline Tag& WithValue(const Aws::String& value) { SetValue(value); return *this;}


inline Tag& WithValue(Aws::String&& value) { SetValue(std::move(value)); return *this;}


inline Tag& WithValue(const char* value) { SetValue(value); return *this;}

private:

Aws::String m_key;
bool m_keyHasBeenSet;

Aws::String m_value;
bool m_valueHasBeenSet;
};

} 
} 
} 
