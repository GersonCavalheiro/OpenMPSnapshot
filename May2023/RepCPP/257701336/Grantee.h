

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/Type.h>
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

class AWS_S3_API Grantee
{
public:
Grantee();
Grantee(const Aws::Utils::Xml::XmlNode& xmlNode);
Grantee& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetDisplayName() const{ return m_displayName; }


inline void SetDisplayName(const Aws::String& value) { m_displayNameHasBeenSet = true; m_displayName = value; }


inline void SetDisplayName(Aws::String&& value) { m_displayNameHasBeenSet = true; m_displayName = std::move(value); }


inline void SetDisplayName(const char* value) { m_displayNameHasBeenSet = true; m_displayName.assign(value); }


inline Grantee& WithDisplayName(const Aws::String& value) { SetDisplayName(value); return *this;}


inline Grantee& WithDisplayName(Aws::String&& value) { SetDisplayName(std::move(value)); return *this;}


inline Grantee& WithDisplayName(const char* value) { SetDisplayName(value); return *this;}



inline const Aws::String& GetEmailAddress() const{ return m_emailAddress; }


inline void SetEmailAddress(const Aws::String& value) { m_emailAddressHasBeenSet = true; m_emailAddress = value; }


inline void SetEmailAddress(Aws::String&& value) { m_emailAddressHasBeenSet = true; m_emailAddress = std::move(value); }


inline void SetEmailAddress(const char* value) { m_emailAddressHasBeenSet = true; m_emailAddress.assign(value); }


inline Grantee& WithEmailAddress(const Aws::String& value) { SetEmailAddress(value); return *this;}


inline Grantee& WithEmailAddress(Aws::String&& value) { SetEmailAddress(std::move(value)); return *this;}


inline Grantee& WithEmailAddress(const char* value) { SetEmailAddress(value); return *this;}



inline const Aws::String& GetID() const{ return m_iD; }


inline void SetID(const Aws::String& value) { m_iDHasBeenSet = true; m_iD = value; }


inline void SetID(Aws::String&& value) { m_iDHasBeenSet = true; m_iD = std::move(value); }


inline void SetID(const char* value) { m_iDHasBeenSet = true; m_iD.assign(value); }


inline Grantee& WithID(const Aws::String& value) { SetID(value); return *this;}


inline Grantee& WithID(Aws::String&& value) { SetID(std::move(value)); return *this;}


inline Grantee& WithID(const char* value) { SetID(value); return *this;}



inline const Type& GetType() const{ return m_type; }


inline void SetType(const Type& value) { m_typeHasBeenSet = true; m_type = value; }


inline void SetType(Type&& value) { m_typeHasBeenSet = true; m_type = std::move(value); }


inline Grantee& WithType(const Type& value) { SetType(value); return *this;}


inline Grantee& WithType(Type&& value) { SetType(std::move(value)); return *this;}



inline const Aws::String& GetURI() const{ return m_uRI; }


inline void SetURI(const Aws::String& value) { m_uRIHasBeenSet = true; m_uRI = value; }


inline void SetURI(Aws::String&& value) { m_uRIHasBeenSet = true; m_uRI = std::move(value); }


inline void SetURI(const char* value) { m_uRIHasBeenSet = true; m_uRI.assign(value); }


inline Grantee& WithURI(const Aws::String& value) { SetURI(value); return *this;}


inline Grantee& WithURI(Aws::String&& value) { SetURI(std::move(value)); return *this;}


inline Grantee& WithURI(const char* value) { SetURI(value); return *this;}

private:

Aws::String m_displayName;
bool m_displayNameHasBeenSet;

Aws::String m_emailAddress;
bool m_emailAddressHasBeenSet;

Aws::String m_iD;
bool m_iDHasBeenSet;

Type m_type;
bool m_typeHasBeenSet;

Aws::String m_uRI;
bool m_uRIHasBeenSet;
};

} 
} 
} 
