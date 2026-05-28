# Security Policy

The system is designed with standard security measures, strictly adhering to privacy protection rules for Health Data.

## Authentication & Authorization
- The entire login and permission granting mechanism is delegated to **Supabase Auth**.
- The API uses the **JWT (JSON Web Token)** standard for security. Frontend files automatically attach a `Bearer Token` when making requests.
- **Row Level Security (RLS)** is enabled on Supabase PostgreSQL to ensure that users are only allowed to view their own measurement history.

## Data Storage & Privacy
- **Videos:** Uploaded video files are temporarily stored in S3 Object Storage (Supabase). After the Celery Worker finishes analyzing and extracting the heart rate, **the video file will be permanently deleted immediately** from the cloud system to ensure image privacy.
- **Webcam Real-time:** Image data from the Webcam is only sent via WebSocket to be processed directly in RAM and return results; **no frames are ever saved** to the hard drive.

## Reporting a Vulnerability
If you discover any security vulnerabilities (such as exposed API Keys in the source code, RLS misconfigurations in Supabase, SQL Injection), please **do not publicly disclose** it as a GitHub Issue.
Please contact the repository administrator directly via email or internal channels so we can resolve it immediately. Thank you!
