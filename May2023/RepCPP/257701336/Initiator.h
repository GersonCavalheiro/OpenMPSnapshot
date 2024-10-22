

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

class AWS_S3_API Initiator
{
public:
Initiator();
Initiator(const Aws::Utils::Xml::XmlNode& xmlNode);
Initiator& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetID() const{ return m_iD; }


inline void SetID(const Aws::String& value) { m_iDHasBeenSet = true; m_iD = value; }


inline void SetID(Aws::String&& value) { m_iDHasBeenSet = true; m_iD = std::move(value); }


inline void SetID(const char* value) { m_iDHasBeenSet = true; m_iD.assign(value); }


inline Initiator& WithID(const Aws::String& value) { SetID(value); return *this;}


inline Initiator& WithID(Aws::String&& value) { SetID(std::move(value)); return *this;}


inline Initiator& WithID(const char* value) { SetID(value); return *this;}



inline const Aws::String& GetDisplayName() const{ return m_displayName; }


inline void SetDisplayName(const Aws::String& value) { m_displayNameHasBeenSet = true; m_displayName = value; }


inline void SetDisplayName(Aws::String&& value) { m_displayNameHasBeenSet = true; m_displayName = std::move(value); }


inline void SetDisplayName(const char* value) { m_displayNameHasBeenSet = true; m_displayName.assign(value); }


inline Initiator& WithDisplayName(const Aws::String& value) { SetDisplayName(value); return *this;}


inline Initiator& WithDisplayName(Aws::String&& value) { SetDisplayName(std::move(value)); return *this;}


inline Initiator& WithDisplayName(const char* value) { SetDisplayName(value); return *this;}

private:

Aws::String m_iD;
bool m_iDHasBeenSet;

Aws::String m_displayName;
bool m_displayNameHasBeenSet;
};

} 
} 
} 
