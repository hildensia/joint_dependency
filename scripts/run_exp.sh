#!/bin/bash

JOINT_DEP_EXP_PY_FOLDER="../"
LOCKBOXES_YAML_FOLDER="../../lockboxes/"

python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_5.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_5.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_5.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_5.yaml 

python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_5.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_5.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_5.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_5.yaml 

python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_5.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_5.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_5.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_5.yaml 

python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_4.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_4.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_4.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_4.yaml 

python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_4.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_4.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_4.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_4.yaml 

python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_4.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_4.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_4.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_4.yaml

python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_6.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_6.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_6.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_6.yaml 

python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_6.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_6.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_6.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_6.yaml 

python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_6.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_6.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_6.yaml 
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 30 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_6.yaml 

