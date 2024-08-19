

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/Grantee.h>
#include <aws/s3/model/BucketLogsPermission.h>
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

class AWS_S3_API TargetGrant
{
public:
TargetGrant();
TargetGrant(const Aws::Utils::Xml::XmlNode& xmlNode);
TargetGrant& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Grantee& GetGrantee() const{ return m_grantee; }


inline void SetGrantee(const Grantee& value) { m_granteeHasBeenSet = true; m_grantee = value; }


inline void SetGrantee(Grantee&& value) { m_granteeHasBeenSet = true; m_grantee = std::move(value); }


inline TargetGrant& WithGrantee(const Grantee& value) { SetGrantee(value); return *this;}


inline TargetGrant& WithGrantee(Grantee&& value) { SetGrantee(std::move(value)); return *this;}



inline const BucketLogsPermission& GetPermission() const{ return m_permission; }


inline void SetPermission(const BucketLogsPermission& value) { m_permissionHasBeenSet = true; m_permission = value; }


inline void SetPermission(BucketLogsPermission&& value) { m_permissionHasBeenSet = true; m_permission = std::move(value); }


inline TargetGrant& WithPermission(const BucketLogsPermission& value) { SetPermission(value); return *this;}


inline TargetGrant& WithPermission(BucketLogsPermission&& value) { SetPermission(std::move(value)); return *this;}

private:

Grantee m_grantee;
bool m_granteeHasBeenSet;

BucketLogsPermission m_permission;
bool m_permissionHasBeenSet;
};

} 
} 
} 
