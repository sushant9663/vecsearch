# Sourced by run_*.sh: resolves dataset file names (SIFT1M or synthetic) into $B $Q $G.
D="${1:-data/sift}"
if [ -f "$D/sift_base.fvecs" ]; then B=$D/sift_base.fvecs; Q=$D/sift_query.fvecs; G=$D/sift_groundtruth.ivecs
else B=$D/base.fvecs; Q=$D/query.fvecs; G=$D/gt.ivecs; fi
mkdir -p results
{ lscpu | grep -E 'Model name|^CPU\(s\)|L2|L3'; echo "date: $(date -u +%F)"; } > results/machine.txt
