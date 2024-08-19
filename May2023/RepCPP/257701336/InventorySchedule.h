

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/InventoryFrequency.h>
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

class AWS_S3_API InventorySchedule
{
public:
InventorySchedule();
InventorySchedule(const Aws::Utils::Xml::XmlNode& xmlNode);
InventorySchedule& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const InventoryFrequency& GetFrequency() const{ return m_frequency; }


inline void SetFrequency(const InventoryFrequency& value) { m_frequencyHasBeenSet = true; m_frequency = value; }


inline void SetFrequency(InventoryFrequency&& value) { m_frequencyHasBeenSet = true; m_frequency = std::move(value); }


inline InventorySchedule& WithFrequency(const InventoryFrequency& value) { SetFrequency(value); return *this;}


inline InventorySchedule& WithFrequency(InventoryFrequency&& value) { SetFrequency(std::move(value)); return *this;}

private:

InventoryFrequency m_frequency;
bool m_frequencyHasBeenSet;
};

} 
} 
} 
