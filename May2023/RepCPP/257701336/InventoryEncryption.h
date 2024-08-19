

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/SSES3.h>
#include <aws/s3/model/SSEKMS.h>
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


class AWS_S3_API InventoryEncryption
{
public:
InventoryEncryption();
InventoryEncryption(const Aws::Utils::Xml::XmlNode& xmlNode);
InventoryEncryption& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const SSES3& GetSSES3() const{ return m_sSES3; }


inline void SetSSES3(const SSES3& value) { m_sSES3HasBeenSet = true; m_sSES3 = value; }


inline void SetSSES3(SSES3&& value) { m_sSES3HasBeenSet = true; m_sSES3 = std::move(value); }


inline InventoryEncryption& WithSSES3(const SSES3& value) { SetSSES3(value); return *this;}


inline InventoryEncryption& WithSSES3(SSES3&& value) { SetSSES3(std::move(value)); return *this;}



inline const SSEKMS& GetSSEKMS() const{ return m_sSEKMS; }


inline void SetSSEKMS(const SSEKMS& value) { m_sSEKMSHasBeenSet = true; m_sSEKMS = value; }


inline void SetSSEKMS(SSEKMS&& value) { m_sSEKMSHasBeenSet = true; m_sSEKMS = std::move(value); }


inline InventoryEncryption& WithSSEKMS(const SSEKMS& value) { SetSSEKMS(value); return *this;}


inline InventoryEncryption& WithSSEKMS(SSEKMS&& value) { SetSSEKMS(std::move(value)); return *this;}

private:

SSES3 m_sSES3;
bool m_sSES3HasBeenSet;

SSEKMS m_sSEKMS;
bool m_sSEKMSHasBeenSet;
};

} 
} 
} 
