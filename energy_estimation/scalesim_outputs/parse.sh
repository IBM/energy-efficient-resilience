#!/bin/bash
awk -F ',' '{sum+=$7}END{print sum/8}' $1/DETAILED_ACCESS_REPORT.csv
