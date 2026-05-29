#!/bin/bash
# setup.sh — Idempotent one-time setup for Pythia cluster access.
#
# Requires: common.sh sourced first (provides PYTHIA_HOST, BLL01_IP, etc.)
#
# Usage:
#   source "${PYTHIA_DIR}/lib/common.sh"
#   source "${PYTHIA_DIR}/lib/setup.sh"
#   pythia_setup

pythia_setup() {
    info "Checking one-time setup..."

    # 1. Authorize pythia's SSH key on bll01 (so compute nodes can rsync back)
    local pythia_pubkey
    pythia_pubkey=$(ssh "${PYTHIA_HOST}" "cat ~/.ssh/id_ecdsa_booth.pub" 2>/dev/null) \
        || error "Cannot read pythia SSH public key. Is ~/.ssh/id_ecdsa_booth.pub present?"

    if ! grep -qF "${pythia_pubkey}" ~/.ssh/authorized_keys 2>/dev/null; then
        info "Adding pythia's SSH key to bll01 authorized_keys..."
        echo "${pythia_pubkey}" >> ~/.ssh/authorized_keys
        chmod 600 ~/.ssh/authorized_keys
    else
        info "Pythia SSH key already authorized on bll01."
    fi

    # 2. SSH config on pythia so compute nodes can reach bll01
    if ! ssh "${PYTHIA_HOST}" "grep -q 'Host bll01' ~/.ssh/config 2>/dev/null"; then
        info "Creating SSH config entry for bll01 on pythia..."
        ssh "${PYTHIA_HOST}" "cat >> ~/.ssh/config && chmod 600 ~/.ssh/config" <<EOF

Host bll01
  HostName ${BLL01_IP}
  User ${BLL01_USER}
  IdentityFile ~/.ssh/id_ecdsa_booth
  StrictHostKeyChecking no
EOF
    else
        info "SSH config for bll01 already exists on pythia."
    fi

    # 3. HuggingFace token on pythia (NFS home, copied to /hpc_temp/ by jobs)
    local hf_cache="~/.cache/huggingface"
    if [ -n "${HF_TOKEN}" ]; then
        local remote_token
        remote_token=$(ssh "${PYTHIA_HOST}" "cat ${hf_cache}/token 2>/dev/null" || true)
        if [ "${remote_token}" != "${HF_TOKEN}" ]; then
            info "Setting up HuggingFace token on pythia..."
            ssh "${PYTHIA_HOST}" "mkdir -p ${hf_cache} && echo '${HF_TOKEN}' > ${hf_cache}/token"
        else
            info "HuggingFace token already on pythia."
        fi
    else
        warn "HF_TOKEN not set in .env — gated models (e.g. Llama) won't download."
    fi

    # 4. Install uv on pythia (NFS home, accessible from compute nodes)
    local uv_bin="~/.local/bin/uv"
    if ! ssh "${PYTHIA_HOST}" "${uv_bin} --version" &>/dev/null; then
        info "Installing uv on pythia..."
        ssh "${PYTHIA_HOST}" "curl -LsSf https://astral.sh/uv/install.sh | sh"
    else
        info "uv already installed on pythia."
    fi
}
