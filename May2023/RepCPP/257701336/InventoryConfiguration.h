

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/InventoryDestination.h>
#include <aws/s3/model/InventoryFilter.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/InventoryIncludedObjectVersions.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/InventorySchedule.h>
#include <aws/s3/model/InventoryOptionalField.h>
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

class AWS_S3_API InventoryConfiguration
{
public:
InventoryConfiguration();
InventoryConfiguration(const Aws::Utils::Xml::XmlNode& xmlNode);
InventoryConfiguration& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const InventoryDestination& GetDestination() const{ return m_destination; }


inline void SetDestination(const InventoryDestination& value) { m_destinationHasBeenSet = true; m_destination = value; }


inline void SetDestination(InventoryDestination&& value) { m_destinationHasBeenSet = true; m_destination = std::move(value); }


inline InventoryConfiguration& WithDestination(const InventoryDestination& value) { SetDestination(value); return *this;}


inline InventoryConfiguration& WithDestination(InventoryDestination&& value) { SetDestination(std::move(value)); return *this;}



inline bool GetIsEnabled() const{ return m_isEnabled; }


inline void SetIsEnabled(bool value) { m_isEnabledHasBeenSet = true; m_isEnabled = value; }


inline InventoryConfiguration& WithIsEnabled(bool value) { SetIsEnabled(value); return *this;}



inline const InventoryFilter& GetFilter() const{ return m_filter; }


inline void SetFilter(const InventoryFilter& value) { m_filterHasBeenSet = true; m_filter = value; }


inline void SetFilter(InventoryFilter&& value) { m_filterHasBeenSet = true; m_filter = std::move(value); }


inline InventoryConfiguration& WithFilter(const InventoryFilter& value) { SetFilter(value); return *this;}


inline InventoryConfiguration& WithFilter(InventoryFilter&& value) { SetFilter(std::move(value)); return *this;}



inline const Aws::String& GetId() const{ return m_id; }


inline void SetId(const Aws::String& value) { m_idHasBeenSet = true; m_id = value; }


inline void SetId(Aws::String&& value) { m_idHasBeenSet = true; m_id = std::move(value); }


inline void SetId(const char* value) { m_idHasBeenSet = true; m_id.assign(value); }


inline InventoryConfiguration& WithId(const Aws::String& value) { SetId(value); return *this;}


inline InventoryConfiguration& WithId(Aws::String&& value) { SetId(std::move(value)); return *this;}


inline InventoryConfiguration& WithId(const char* value) { SetId(value); return *this;}



inline const InventoryIncludedObjectVersions& GetIncludedObjectVersions() const{ return m_includedObjectVersions; }


inline void SetIncludedObjectVersions(const InventoryIncludedObjectVersions& value) { m_includedObjectVersionsHasBeenSet = true; m_includedObjectVersions = value; }


inline void SetIncludedObjectVersions(InventoryIncludedObjectVersions&& value) { m_includedObjectVersionsHasBeenSet = true; m_includedObjectVersions = std::move(value); }


inline InventoryConfiguration& WithIncludedObjectVersions(const InventoryIncludedObjectVersions& value) { SetIncludedObjectVersions(value); return *this;}


inline InventoryConfiguration& WithIncludedObjectVersions(InventoryIncludedObjectVersions&& value) { SetIncludedObjectVersions(std::move(value)); return *this;}



inline const Aws::Vector<InventoryOptionalField>& GetOptionalFields() const{ return m_optionalFields; }


inline void SetOptionalFields(const Aws::Vector<InventoryOptionalField>& value) { m_optionalFieldsHasBeenSet = true; m_optionalFields = value; }


inline void SetOptionalFields(Aws::Vector<InventoryOptionalField>&& value) { m_optionalFieldsHasBeenSet = true; m_optionalFields = std::move(value); }


inline InventoryConfiguration& WithOptionalFields(const Aws::Vector<InventoryOptionalField>& value) { SetOptionalFields(value); return *this;}


inline InventoryConfiguration& WithOptionalFields(Aws::Vector<InventoryOptionalField>&& value) { SetOptionalFields(std::move(value)); return *this;}


inline InventoryConfiguration& AddOptionalFields(const InventoryOptionalField& value) { m_optionalFieldsHasBeenSet = true; m_optionalFields.push_back(value); return *this; }


inline InventoryConfiguration& AddOptionalFields(InventoryOptionalField&& value) { m_optionalFieldsHasBeenSet = true; m_optionalFields.push_back(std::move(value)); return *this; }



inline const InventorySchedule& GetSchedule() const{ return m_schedule; }


inline void SetSchedule(const InventorySchedule& value) { m_scheduleHasBeenSet = true; m_schedule = value; }


inline void SetSchedule(InventorySchedule&& value) { m_scheduleHasBeenSet = true; m_schedule = std::move(value); }


inline InventoryConfiguration& WithSchedule(const InventorySchedule& value) { SetSchedule(value); return *this;}


inline InventoryConfiguration& WithSchedule(InventorySchedule&& value) { SetSchedule(std::move(value)); return *this;}

private:

InventoryDestination m_destination;
bool m_destinationHasBeenSet;

bool m_isEnabled;
bool m_isEnabledHasBeenSet;

InventoryFilter m_filter;
bool m_filterHasBeenSet;

Aws::String m_id;
bool m_idHasBeenSet;

InventoryIncludedObjectVersions m_includedObjectVersions;
bool m_includedObjectVersionsHasBeenSet;

Aws::Vector<InventoryOptionalField> m_optionalFields;
bool m_optionalFieldsHasBeenSet;

InventorySchedule m_schedule;
bool m_scheduleHasBeenSet;
};

} 
} 
} 
