

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/DateTime.h>
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

class AWS_S3_API LifecycleExpiration
{
public:
LifecycleExpiration();
LifecycleExpiration(const Aws::Utils::Xml::XmlNode& xmlNode);
LifecycleExpiration& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::Utils::DateTime& GetDate() const{ return m_date; }


inline void SetDate(const Aws::Utils::DateTime& value) { m_dateHasBeenSet = true; m_date = value; }


inline void SetDate(Aws::Utils::DateTime&& value) { m_dateHasBeenSet = true; m_date = std::move(value); }


inline LifecycleExpiration& WithDate(const Aws::Utils::DateTime& value) { SetDate(value); return *this;}


inline LifecycleExpiration& WithDate(Aws::Utils::DateTime&& value) { SetDate(std::move(value)); return *this;}



inline int GetDays() const{ return m_days; }


inline void SetDays(int value) { m_daysHasBeenSet = true; m_days = value; }


inline LifecycleExpiration& WithDays(int value) { SetDays(value); return *this;}



inline bool GetExpiredObjectDeleteMarker() const{ return m_expiredObjectDeleteMarker; }


inline void SetExpiredObjectDeleteMarker(bool value) { m_expiredObjectDeleteMarkerHasBeenSet = true; m_expiredObjectDeleteMarker = value; }


inline LifecycleExpiration& WithExpiredObjectDeleteMarker(bool value) { SetExpiredObjectDeleteMarker(value); return *this;}

private:

Aws::Utils::DateTime m_date;
bool m_dateHasBeenSet;

int m_days;
bool m_daysHasBeenSet;

bool m_expiredObjectDeleteMarker;
bool m_expiredObjectDeleteMarkerHasBeenSet;
};

} 
} 
} 
