#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?usage: freeze_doc_to_lora_repo_chunks.sh OUTPUT_ROOT [FREEZE_ID]}"
FREEZE_ID="${2:-frozen_$(date -u +%Y%m%dT%H%M%SZ)}"
EXPECTED_COUNT="${EXPECTED_COUNT:-833}"
ORIGINAL_COUNT="${ORIGINAL_COUNT:-852}"
FREEZES_DIR="${ROOT}/freezes"
FINAL_DIR="${FREEZES_DIR}/${FREEZE_ID}"
TMP_DIR="${FREEZES_DIR}/.${FREEZE_ID}.tmp.$$"

if [[ -e "${FINAL_DIR}" ]]; then
  echo "ERROR: freeze already exists: ${FINAL_DIR}" >&2
  exit 1
fi

mkdir -p "${TMP_DIR}"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

find "${ROOT}/repositories" -mindepth 2 -maxdepth 2 -type f -name audit.json -print0 \
  | sort -z > "${TMP_DIR}/audits.nul"

count=0
: > "${TMP_DIR}/repositories.jsonl"
: > "${TMP_DIR}/repository_ids.txt"
while IFS= read -r -d '' audit; do
  repo_dir="$(dirname "${audit}")"
  jq -e '.status == "complete"' "${audit}" >/dev/null
  [[ -f "${repo_dir}/chunks.parquet" ]]
  [[ -f "${repo_dir}/snapshots.parquet" ]]
  jq -c '{repo_id, status, chunks_parquet, snapshots_parquet, counters, config}' \
    "${audit}" >> "${TMP_DIR}/repositories.jsonl"
  jq -r '.repo_id' "${audit}" >> "${TMP_DIR}/repository_ids.txt"
  count=$((count + 1))
done < "${TMP_DIR}/audits.nul"

sort -u -o "${TMP_DIR}/repository_ids.txt" "${TMP_DIR}/repository_ids.txt"
unique_count="$(wc -l < "${TMP_DIR}/repository_ids.txt")"
if [[ "${count}" != "${EXPECTED_COUNT}" || "${unique_count}" != "${EXPECTED_COUNT}" ]]; then
  echo "ERROR: expected ${EXPECTED_COUNT} complete unique repositories, found audits=${count} unique=${unique_count}" >&2
  exit 1
fi

ids_sha256="$(sha256sum "${TMP_DIR}/repository_ids.txt" | awk '{print $1}')"
records_sha256="$(sha256sum "${TMP_DIR}/repositories.jsonl" | awk '{print $1}')"
frozen_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

jq -n \
  --arg freeze_id "${FREEZE_ID}" \
  --arg frozen_utc "${frozen_utc}" \
  --arg output_root "${ROOT}" \
  --arg ids_sha256 "${ids_sha256}" \
  --arg records_sha256 "${records_sha256}" \
  --argjson completed_repositories "${count}" \
  --argjson original_repositories "${ORIGINAL_COUNT}" \
  --argjson unavailable_repositories "$((ORIGINAL_COUNT - count))" \
  '{
    freeze_id: $freeze_id,
    status: "frozen",
    frozen_utc: $frozen_utc,
    output_root: $output_root,
    completed_repositories: $completed_repositories,
    original_repositories: $original_repositories,
    unavailable_repositories: $unavailable_repositories,
    repository_ids_file: "repository_ids.txt",
    repository_ids_sha256: $ids_sha256,
    repository_records_file: "repositories.jsonl",
    repository_records_sha256: $records_sha256,
    deliberately_excluded_or_stopped: [
      "aws-cloudformation/cfn-lint",
      "tobymao/sqlglot",
      "pymedusa/Medusa",
      "WeblateOrg/weblate",
      "biolab/orange3-text",
      "python/mypy",
      "ietf-tools/datatracker",
      "sktime/sktime",
      "googleapis/google-cloud-python"
    ],
    note: "Training must use only repository IDs in repository_ids.txt. Ten additional repositories were not reached after the final three stopped shards."
  }' > "${TMP_DIR}/manifest.json"

rm "${TMP_DIR}/audits.nul"
mv "${TMP_DIR}" "${FINAL_DIR}"
trap - EXIT

printf '%s\n' "${FINAL_DIR}"
