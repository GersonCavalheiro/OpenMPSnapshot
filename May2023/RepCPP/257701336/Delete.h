

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/ObjectIdentifier.h>
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

class AWS_S3_API Delete
{
public:
Delete();
Delete(const Aws::Utils::Xml::XmlNode& xmlNode);
Delete& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::Vector<ObjectIdentifier>& GetObjects() const{ return m_objects; }


inline void SetObjects(const Aws::Vector<ObjectIdentifier>& value) { m_objectsHasBeenSet = true; m_objects = value; }


inline void SetObjects(Aws::Vector<ObjectIdentifier>&& value) { m_objectsHasBeenSet = true; m_objects = std::move(value); }


inline Delete& WithObjects(const Aws::Vector<ObjectIdentifier>& value) { SetObjects(value); return *this;}


inline Delete& WithObjects(Aws::Vector<ObjectIdentifier>&& value) { SetObjects(std::move(value)); return *this;}


inline Delete& AddObjects(const ObjectIdentifier& value) { m_objectsHasBeenSet = true; m_objects.push_back(value); return *this; }


inline Delete& AddObjects(ObjectIdentifier&& value) { m_objectsHasBeenSet = true; m_objects.push_back(std::move(value)); return *this; }



inline bool GetQuiet() const{ return m_quiet; }


inline void SetQuiet(bool value) { m_quietHasBeenSet = true; m_quiet = value; }


inline Delete& WithQuiet(bool value) { SetQuiet(value); return *this;}

private:

Aws::Vector<ObjectIdentifier> m_objects;
bool m_objectsHasBeenSet;

bool m_quiet;
bool m_quietHasBeenSet;
};

} 
} 
} 
