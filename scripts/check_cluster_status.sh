#!/bin/bash
squeue -u swathy
ls -lt ~/self-questioning-vlms/slurm-*.out 2>/dev/null | head -5
ls -lt ~/self-questioning-vlms/output/self-questioning-aokvqa/ 2>/dev/null | head -10
tail -50 $(ls -t ~/self-questioning-vlms/slurm-*.out 2>/dev/null | head -1) 2>/dev/null
cat ~/self-questioning-vlms/output/eval_aokvqa_direct.json 2>/dev/null | head -20 || echo "No direct eval"
cat ~/self-questioning-vlms/output/eval_aokvqa_sq.json 2>/dev/null | head -20 || echo "No SQ eval"
