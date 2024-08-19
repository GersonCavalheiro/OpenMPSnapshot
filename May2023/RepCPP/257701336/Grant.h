

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/Grantee.h>
#include <aws/s3/model/Permission.h>
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

class AWS_S3_API Grant
{
public:
Grant();
Grant(const Aws::Utils::Xml::XmlNode& xmlNode);
Grant& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Grantee& GetGrantee() const{ return m_grantee; }


inline void SetGrantee(const Grantee& value) { m_granteeHasBeenSet = true; m_grantee = value; }


inline void SetGrantee(Grantee&& value) { m_granteeHasBeenSet = true; m_grantee = std::move(value); }


inline Grant& WithGrantee(const Grantee& value) { SetGrantee(value); return *this;}


inline Grant& WithGrantee(Grantee&& value) { SetGrantee(std::move(value)); return *this;}



inline const Permission& GetPermission() const{ return m_permission; }


inline void SetPermission(const Permission& value) { m_permissionHasBeenSet = true; m_permission = value; }


inline void SetPermission(Permission&& value) { m_permissionHasBeenSet = true; m_permission = std::move(value); }


inline Grant& WithPermission(const Permission& value) { SetPermission(value); return *this;}


inline Grant& WithPermission(Permission&& value) { SetPermission(std::move(value)); return *this;}

private:

Grantee m_grantee;
bool m_granteeHasBeenSet;

Permission m_permission;
bool m_permissionHasBeenSet;
};

} 
} 
} 
