#!/usr/bin/env bash
export PYTHONPATH=/app/jenkins-slave/workspace/ZCPZ/zcpz/ZCPZ_UT_Jacoco


find  .-name "*Test.py" -print | while read f; do echo "$f"
      ###
      python -m coverage run "$f"
      python -m coverage combine
      python -m coverage xml -o coverage.xml
      ###
done