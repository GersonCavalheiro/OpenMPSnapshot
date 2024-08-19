

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/OwnerOverride.h>
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


class AWS_S3_API AccessControlTranslation
{
public:
AccessControlTranslation();
AccessControlTranslation(const Aws::Utils::Xml::XmlNode& xmlNode);
AccessControlTranslation& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const OwnerOverride& GetOwner() const{ return m_owner; }


inline void SetOwner(const OwnerOverride& value) { m_ownerHasBeenSet = true; m_owner = value; }


inline void SetOwner(OwnerOverride&& value) { m_ownerHasBeenSet = true; m_owner = std::move(value); }


inline AccessControlTranslation& WithOwner(const OwnerOverride& value) { SetOwner(value); return *this;}


inline AccessControlTranslation& WithOwner(OwnerOverride&& value) { SetOwner(std::move(value)); return *this;}

private:

OwnerOverride m_owner;
bool m_ownerHasBeenSet;
};

} 
} 
} 
