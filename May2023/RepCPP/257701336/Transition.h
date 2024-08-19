

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/DateTime.h>
#include <aws/s3/model/TransitionStorageClass.h>
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

class AWS_S3_API Transition
{
public:
Transition();
Transition(const Aws::Utils::Xml::XmlNode& xmlNode);
Transition& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::Utils::DateTime& GetDate() const{ return m_date; }


inline void SetDate(const Aws::Utils::DateTime& value) { m_dateHasBeenSet = true; m_date = value; }


inline void SetDate(Aws::Utils::DateTime&& value) { m_dateHasBeenSet = true; m_date = std::move(value); }


inline Transition& WithDate(const Aws::Utils::DateTime& value) { SetDate(value); return *this;}


inline Transition& WithDate(Aws::Utils::DateTime&& value) { SetDate(std::move(value)); return *this;}



inline int GetDays() const{ return m_days; }


inline void SetDays(int value) { m_daysHasBeenSet = true; m_days = value; }


inline Transition& WithDays(int value) { SetDays(value); return *this;}



inline const TransitionStorageClass& GetStorageClass() const{ return m_storageClass; }


inline void SetStorageClass(const TransitionStorageClass& value) { m_storageClassHasBeenSet = true; m_storageClass = value; }


inline void SetStorageClass(TransitionStorageClass&& value) { m_storageClassHasBeenSet = true; m_storageClass = std::move(value); }


inline Transition& WithStorageClass(const TransitionStorageClass& value) { SetStorageClass(value); return *this;}


inline Transition& WithStorageClass(TransitionStorageClass&& value) { SetStorageClass(std::move(value)); return *this;}

private:

Aws::Utils::DateTime m_date;
bool m_dateHasBeenSet;

int m_days;
bool m_daysHasBeenSet;

TransitionStorageClass m_storageClass;
bool m_storageClassHasBeenSet;
};

} 
} 
} 
