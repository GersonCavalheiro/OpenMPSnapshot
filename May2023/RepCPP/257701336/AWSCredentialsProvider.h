


#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/UnreferencedParam.h>
#include <aws/core/utils/DateTime.h>
#include <aws/core/utils/memory/stl/AWSMap.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/internal/AWSHttpResourceClient.h>
#include <memory>
#include <mutex>

namespace Aws
{
namespace Config
{
class AWSProfileConfigLoader;
class EC2InstanceProfileConfigLoader;
}

namespace Auth
{
static int REFRESH_THRESHOLD = 1000 * 60 * 15;
static int EXPIRATION_GRACE_PERIOD = 5 * 1000;


class AWS_CORE_API AWSCredentials
{
public:
AWSCredentials()
{
}


AWSCredentials(const Aws::String& accessKeyId, const Aws::String& secretKey, const Aws::String& sessionToken = "") :
m_accessKeyId(accessKeyId), m_secretKey(secretKey), m_sessionToken(sessionToken)
{
}


inline const Aws::String& GetAWSAccessKeyId() const
{
return m_accessKeyId;
}


inline const Aws::String& GetAWSSecretKey() const
{
return m_secretKey;
}


inline const Aws::String& GetSessionToken() const
{
return m_sessionToken;
}


inline void SetAWSAccessKeyId(const Aws::String& accessKeyId)
{
m_accessKeyId = accessKeyId;
}


inline void SetAWSSecretKey(const Aws::String& secretKey)
{
m_secretKey = secretKey;
}


inline void SetSessionToken(const Aws::String& sessionToken)
{
m_sessionToken = sessionToken;
}


inline void SetAWSAccessKeyId(const char* accessKeyId)
{
m_accessKeyId = accessKeyId;
}


inline void SetAWSSecretKey(const char* secretKey)
{
m_secretKey = secretKey;
}


inline void SetSessionToken(const char* sessionToken)
{
m_sessionToken = sessionToken;
}

private:
Aws::String m_accessKeyId;
Aws::String m_secretKey;
Aws::String m_sessionToken;
};


class AWS_CORE_API AWSCredentialsProvider
{
public:

AWSCredentialsProvider() : m_lastLoadedMs(0)
{
}

virtual ~AWSCredentialsProvider() = default;


virtual AWSCredentials GetAWSCredentials() = 0;

protected:

virtual bool IsTimeToRefresh(long reloadFrequency);

private:
long long m_lastLoadedMs;
};


class AWS_CORE_API AnonymousAWSCredentialsProvider : public AWSCredentialsProvider
{
public:

inline AWSCredentials GetAWSCredentials() override { return AWSCredentials("", ""); }
};


class AWS_CORE_API SimpleAWSCredentialsProvider : public AWSCredentialsProvider
{
public:

inline SimpleAWSCredentialsProvider(const Aws::String& awsAccessKeyId, const Aws::String& awsSecretAccessKey, const Aws::String& sessionToken = "")
: m_accessKeyId(awsAccessKeyId), m_secretAccessKey(awsSecretAccessKey), m_sessionToken(sessionToken)
{ }


inline SimpleAWSCredentialsProvider(const AWSCredentials& credentials)
: m_accessKeyId(credentials.GetAWSAccessKeyId()), m_secretAccessKey(credentials.GetAWSSecretKey()),
m_sessionToken(credentials.GetSessionToken())
{ }


inline AWSCredentials GetAWSCredentials() override
{
return AWSCredentials(m_accessKeyId, m_secretAccessKey, m_sessionToken);
}

private:
Aws::String m_accessKeyId;
Aws::String m_secretAccessKey;
Aws::String m_sessionToken;
};


class AWS_CORE_API EnvironmentAWSCredentialsProvider : public AWSCredentialsProvider
{
public:

AWSCredentials GetAWSCredentials() override;
};


class AWS_CORE_API ProfileConfigFileAWSCredentialsProvider : public AWSCredentialsProvider
{
public:


ProfileConfigFileAWSCredentialsProvider(long refreshRateMs = REFRESH_THRESHOLD);


ProfileConfigFileAWSCredentialsProvider(const char* profile, long refreshRateMs = REFRESH_THRESHOLD);


AWSCredentials GetAWSCredentials() override;


static Aws::String GetConfigProfileFilename();


static Aws::String GetCredentialsProfileFilename();


static Aws::String GetProfileDirectory();

private:


void RefreshIfExpired();

Aws::String m_profileToUse;
std::shared_ptr<Aws::Config::AWSProfileConfigLoader> m_configFileLoader;
std::shared_ptr<Aws::Config::AWSProfileConfigLoader> m_credentialsFileLoader;
mutable std::mutex m_reloadMutex;
long m_loadFrequencyMs;
};


class AWS_CORE_API InstanceProfileCredentialsProvider : public AWSCredentialsProvider
{
public:

InstanceProfileCredentialsProvider(long refreshRateMs = REFRESH_THRESHOLD);


InstanceProfileCredentialsProvider(const std::shared_ptr<Aws::Config::EC2InstanceProfileConfigLoader>&, long refreshRateMs = REFRESH_THRESHOLD);


AWSCredentials GetAWSCredentials() override;

private:
void RefreshIfExpired();

std::shared_ptr<Aws::Config::AWSProfileConfigLoader> m_ec2MetadataConfigLoader;
long m_loadFrequencyMs;
mutable std::mutex m_reloadMutex;
};


class AWS_CORE_API TaskRoleCredentialsProvider : public AWSCredentialsProvider
{
public:

TaskRoleCredentialsProvider(const char* resourcePath, long refreshRateMs = REFRESH_THRESHOLD);


TaskRoleCredentialsProvider(const std::shared_ptr<Aws::Internal::ECSCredentialsClient>& client,
long refreshRateMs = REFRESH_THRESHOLD);

AWSCredentials GetAWSCredentials() override;

private:

inline bool ExpiresSoon() 
{
return (m_expirationDate.Millis() - Aws::Utils::DateTime::Now().Millis() < EXPIRATION_GRACE_PERIOD);
}

void RefreshIfExpired();

private:
std::shared_ptr<Aws::Internal::ECSCredentialsClient> m_ecsCredentialsClient;
long m_loadFrequencyMs;
mutable std::mutex m_reloadMutex;
Aws::Utils::DateTime m_expirationDate;
Aws::Auth::AWSCredentials m_credentials;
};

} 
} 
