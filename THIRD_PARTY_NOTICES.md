# Source and dependency notices

Dependencies are listed in [pyproject.toml](pyproject.toml), with resolved versions in [uv.lock](uv.lock). They are installed separately and retain their own licenses and notices.

ACS survey records come from the U.S. Census Bureau; ACSPubCov and ACSMobility use Folktables to download them. CPS earnings records are distributed by the National Bureau of Economic Research. Adult is downloaded through OpenML. This repository does not redistribute these source datasets; their source terms and attribution apply. Dataset construction is described in the [protocol](paper/protocol.md#datasets).
