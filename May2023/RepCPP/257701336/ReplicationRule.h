

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/ReplicationRuleStatus.h>
#include <aws/s3/model/SourceSelectionCriteria.h>
#include <aws/s3/model/Destination.h>
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


class AWS_S3_API ReplicationRule
{
public:
ReplicationRule();
ReplicationRule(const Aws::Utils::Xml::XmlNode& xmlNode);
ReplicationRule& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetID() const{ return m_iD; }


inline void SetID(const Aws::String& value) { m_iDHasBeenSet = true; m_iD = value; }


inline void SetID(Aws::String&& value) { m_iDHasBeenSet = true; m_iD = std::move(value); }


inline void SetID(const char* value) { m_iDHasBeenSet = true; m_iD.assign(value); }


inline ReplicationRule& WithID(const Aws::String& value) { SetID(value); return *this;}


inline ReplicationRule& WithID(Aws::String&& value) { SetID(std::move(value)); return *this;}


inline ReplicationRule& WithID(const char* value) { SetID(value); return *this;}



inline const Aws::String& GetPrefix() const{ return m_prefix; }


inline void SetPrefix(const Aws::String& value) { m_prefixHasBeenSet = true; m_prefix = value; }


inline void SetPrefix(Aws::String&& value) { m_prefixHasBeenSet = true; m_prefix = std::move(value); }


inline void SetPrefix(const char* value) { m_prefixHasBeenSet = true; m_prefix.assign(value); }


inline ReplicationRule& WithPrefix(const Aws::String& value) { SetPrefix(value); return *this;}


inline ReplicationRule& WithPrefix(Aws::String&& value) { SetPrefix(std::move(value)); return *this;}


inline ReplicationRule& WithPrefix(const char* value) { SetPrefix(value); return *this;}



inline const ReplicationRuleStatus& GetStatus() const{ return m_status; }


inline void SetStatus(const ReplicationRuleStatus& value) { m_statusHasBeenSet = true; m_status = value; }


inline void SetStatus(ReplicationRuleStatus&& value) { m_statusHasBeenSet = true; m_status = std::move(value); }


inline ReplicationRule& WithStatus(const ReplicationRuleStatus& value) { SetStatus(value); return *this;}


inline ReplicationRule& WithStatus(ReplicationRuleStatus&& value) { SetStatus(std::move(value)); return *this;}



inline const SourceSelectionCriteria& GetSourceSelectionCriteria() const{ return m_sourceSelectionCriteria; }


inline void SetSourceSelectionCriteria(const SourceSelectionCriteria& value) { m_sourceSelectionCriteriaHasBeenSet = true; m_sourceSelectionCriteria = value; }


inline void SetSourceSelectionCriteria(SourceSelectionCriteria&& value) { m_sourceSelectionCriteriaHasBeenSet = true; m_sourceSelectionCriteria = std::move(value); }


inline ReplicationRule& WithSourceSelectionCriteria(const SourceSelectionCriteria& value) { SetSourceSelectionCriteria(value); return *this;}


inline ReplicationRule& WithSourceSelectionCriteria(SourceSelectionCriteria&& value) { SetSourceSelectionCriteria(std::move(value)); return *this;}



inline const Destination& GetDestination() const{ return m_destination; }


inline void SetDestination(const Destination& value) { m_destinationHasBeenSet = true; m_destination = value; }


inline void SetDestination(Destination&& value) { m_destinationHasBeenSet = true; m_destination = std::move(value); }


inline ReplicationRule& WithDestination(const Destination& value) { SetDestination(value); return *this;}


inline ReplicationRule& WithDestination(Destination&& value) { SetDestination(std::move(value)); return *this;}

private:

Aws::String m_iD;
bool m_iDHasBeenSet;

Aws::String m_prefix;
bool m_prefixHasBeenSet;

ReplicationRuleStatus m_status;
bool m_statusHasBeenSet;

SourceSelectionCriteria m_sourceSelectionCriteria;
bool m_sourceSelectionCriteriaHasBeenSet;

Destination m_destination;
bool m_destinationHasBeenSet;
};

} 
} 
} 
