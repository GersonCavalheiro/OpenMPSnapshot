

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/CompletedPart.h>
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

class AWS_S3_API CompletedMultipartUpload
{
public:
CompletedMultipartUpload();
CompletedMultipartUpload(const Aws::Utils::Xml::XmlNode& xmlNode);
CompletedMultipartUpload& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::Vector<CompletedPart>& GetParts() const{ return m_parts; }


inline void SetParts(const Aws::Vector<CompletedPart>& value) { m_partsHasBeenSet = true; m_parts = value; }


inline void SetParts(Aws::Vector<CompletedPart>&& value) { m_partsHasBeenSet = true; m_parts = std::move(value); }


inline CompletedMultipartUpload& WithParts(const Aws::Vector<CompletedPart>& value) { SetParts(value); return *this;}


inline CompletedMultipartUpload& WithParts(Aws::Vector<CompletedPart>&& value) { SetParts(std::move(value)); return *this;}


inline CompletedMultipartUpload& AddParts(const CompletedPart& value) { m_partsHasBeenSet = true; m_parts.push_back(value); return *this; }


inline CompletedMultipartUpload& AddParts(CompletedPart&& value) { m_partsHasBeenSet = true; m_parts.push_back(std::move(value)); return *this; }

private:

Aws::Vector<CompletedPart> m_parts;
bool m_partsHasBeenSet;
};

} 
} 
} 
