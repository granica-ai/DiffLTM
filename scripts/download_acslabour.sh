#!/usr/bin/env bash
# Download the 52 Census 2018 five-year ACS person files (50 states, DC, Puerto Rico)
# used to build ACSLabour. Files go where the ACSLabour loader looks by default:
#   ${DIFFLTM_DATA:-data}/raw/acslabour/zips
# or to the directory given as the first argument (then set DIFFLTM_ACSLABOUR_RAW to it).
# Existing files are kept, so the script can be re-run after an interruption.
set -euo pipefail

dest=${1:-${DIFFLTM_DATA:-data}/raw/acslabour/zips}
base=https://www2.census.gov/programs-surveys/acs/data/pums/2018/5-Year
states="al ak az ar ca co ct de dc fl ga hi id il in ia ks ky la me md ma mi mn ms mo
        mt ne nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy pr"

mkdir -p "$dest"
for s in $states; do
  file="$dest/csv_p$s.zip"
  if [[ -s "$file" ]]; then continue; fi
  echo "Downloading csv_p$s.zip"
  curl -fL --retry 3 -o "$file.part" "$base/csv_p$s.zip"
  mv "$file.part" "$file"
done

count=$(ls "$dest"/csv_p*.zip | wc -l)
if [[ "$count" -ne 52 ]]; then
  echo "Expected 52 archives in $dest, found $count" >&2
  exit 1
fi
echo "All 52 ACSLabour archives are in $dest"
