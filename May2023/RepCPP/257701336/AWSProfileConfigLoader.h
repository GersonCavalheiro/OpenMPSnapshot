

#pragma once

#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/utils/DateTime.h>

namespace Aws
{
namespace Internal
{
class EC2MetadataClient;
}

namespace Config
{

class Profile
{
public:
inline const Aws::String& GetName() const { return m_name; }
inline void SetName(const Aws::String& value) { m_name = value; }
inline const Aws::Auth::AWSCredentials& GetCredentials() const { return m_credentials; }
inline void SetCredentials(const Aws::Auth::AWSCredentials& value) { m_credentials = value; }
inline const Aws::String& GetRegion() const { return m_region; }
inline void SetRegion(const Aws::String& value) { m_region = value; }
inline const Aws::String& GetRoleArn() const { return m_roleArn; }
inline void SetRoleArn(const Aws::String& value) { m_roleArn = value; }
inline const Aws::String& GetSourceProfile() const { return m_sourceProfile; }
inline void SetSourceProfile(const Aws::String& value ) { m_sourceProfile = value; }
inline void SetAllKeyValPairs(const Aws::Map<Aws::String, Aws::String>& map) { m_allKeyValPairs = map; }
inline const Aws::String GetValue(const Aws::String& key) 
{
auto iter = m_allKeyValPairs.find(key);
if (iter == m_allKeyValPairs.end()) return "";
return iter->second;
}

private:
Aws::String m_name;
Aws::String m_region;
Aws::Auth::AWSCredentials m_credentials;
Aws::String m_roleArn;
Aws::String m_sourceProfile;

Aws::Map<Aws::String, Aws::String> m_allKeyValPairs;
};


class AWS_CORE_API AWSProfileConfigLoader
{
public:
virtual ~AWSProfileConfigLoader() = default;


bool Load();


bool PersistProfiles(const Aws::Map<Aws::String, Aws::Config::Profile>& profiles);


inline const Aws::Map<Aws::String, Aws::Config::Profile>& GetProfiles() const { return m_profiles; };


inline const Aws::Utils::DateTime& LastLoadTime() const { return m_lastLoadTime; }

protected:

virtual bool LoadInternal() = 0;


virtual bool PersistInternal(const Aws::Map<Aws::String, Aws::Config::Profile>&) { return false; }

Aws::Map<Aws::String, Aws::Config::Profile> m_profiles;
Aws::Utils::DateTime m_lastLoadTime;
};


class AWS_CORE_API AWSConfigFileProfileConfigLoader : public AWSProfileConfigLoader
{
public:

AWSConfigFileProfileConfigLoader(const Aws::String& fileName, bool useProfilePrefix = false);

virtual ~AWSConfigFileProfileConfigLoader() = default;


const Aws::String& GetFileName() const { return m_fileName; }

protected:
virtual bool LoadInternal() override;
virtual bool PersistInternal(const Aws::Map<Aws::String, Aws::Config::Profile>&) override;

private:
Aws::String m_fileName;
bool m_useProfilePrefix;
};

static const char* const INSTANCE_PROFILE_KEY = "InstanceProfile";


class AWS_CORE_API EC2InstanceProfileConfigLoader : public AWSProfileConfigLoader
{
public:

EC2InstanceProfileConfigLoader(const std::shared_ptr<Aws::Internal::EC2MetadataClient>& = nullptr);

virtual ~EC2InstanceProfileConfigLoader() = default;

protected:
virtual bool LoadInternal() override;

private:
std::shared_ptr<Aws::Internal::EC2MetadataClient> m_ec2metadataClient;
};
}
}
