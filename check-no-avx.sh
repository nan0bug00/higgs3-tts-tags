#!/usr/bin/env bash
# Verify a binary (and its ggml shared libraries) contain no AVX-family
# instructions, so they run on the legacy CPUs vast.ai still rents.
#
# ggml's GGML_NATIVE defaults to ON (-march=native). Building on a modern EPYC
# host silently bakes in AVX-512 and the binary SIGILLs elsewhere. The cmake
# flags in the Dockerfile are supposed to prevent that; this checks that they
# actually did, because "the flag was set" is not evidence.
#
# Detection: every AVX/AVX2/AVX-512/FMA/F16C instruction is VEX/EVEX-encoded and
# disassembles with a leading 'v' (vmovups, vaddps, vfmadd132ps, ...), plus the
# AVX-512 mask-register ops (kmovw, kortestw, ...). Matching the mnemonic prefix
# catches all of them without maintaining a list.
set -euo pipefail

TARGETS=()
for arg in "$@"; do
    if [[ -d "$arg" ]]; then
        while IFS= read -r f; do TARGETS+=("$f"); done \
            < <(find "$arg" -type f \( -name '*.so' -o -name '*.so.*' \) | sort)
    elif [[ -f "$arg" ]]; then
        TARGETS+=("$arg")
    fi
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "check-no-avx: no targets given" >&2
    exit 1
fi

# 'verr'/'verw'/'vm*' are the only non-AVX x86 mnemonics starting with v; they
# are privileged/segment ops that never appear in this code, but exclude them
# anyway so a hit is unambiguous.
NOT_AVX='^(verr|verw|vmcall|vmlaunch|vmresume|vmxoff|vmxon|vmread|vmwrite|vmptrld|vmptrst|vmclear|vmfunc|vmmcall|vmload|vmsave|vmrun)$'

status=0
for t in "${TARGETS[@]}"; do
    hits=$(objdump -d --no-show-raw-insn "$t" 2>/dev/null \
        | awk -F'\t' 'NF > 1 { split($2, a, " "); print a[1] }' \
        | grep -E '^(v[a-z0-9]+|k[a-z]+[bwdq])$' \
        | grep -Ev "$NOT_AVX" \
        | sort | uniq -c | sort -rn || true)

    if [[ -n "$hits" ]]; then
        echo "FAIL: AVX-family instructions in $t" >&2
        echo "$hits" | head -20 >&2
        status=1
    else
        echo "OK: no AVX-family instructions in $(basename "$t")"
    fi
done

if [[ $status -ne 0 ]]; then
    echo "" >&2
    echo "This ggml build is not CPU-generic and will SIGILL on older hosts." >&2
    echo "Check the GGML_NATIVE / GGML_AVX* / GGML_FMA / GGML_F16C cmake flags." >&2
    exit 1
fi

echo "check-no-avx: all ${#TARGETS[@]} target(s) clean"
