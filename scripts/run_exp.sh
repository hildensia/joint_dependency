#!/bin/bash

JOINT_DEP_EXP_PY_FOLDER="../"
LOCKBOXES_YAML_FOLDER="../../lockboxes/"

RANDOM_OBJ=1
ENT_OBJ=1
CE_OBJ=0
OSCE_OBJ=0
HEU_OBJ=1

FOUR=1
FIVE=1
SIX=1
REALDOUBLE=0

if [ ${RANDOM_OBJ} -eq 1 ]
then
if [ ${FOUR} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_4.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_4.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_4.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_opened_one_4.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_opened_one_4.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_opened_one_4.yaml
fi
if [ ${FIVE} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_5.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_5.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_5.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_opened_one_5.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_opened_one_5.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_opened_one_5.yaml
fi
if [ ${SIX} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_6.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_6.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_6.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_opened_one_6.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_opened_one_6.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_opened_one_6.yaml
fi
if [ ${REALDOUBLE} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o random --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_icra_2_lock_1.yaml
fi
fi

if [ ${ENT_OBJ} -eq 1 ]
then
if [ ${FOUR} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_4.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_4.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_4.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_opened_one_4.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_opened_one_4.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_opened_one_4.yaml
fi
if [ ${FIVE} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_5.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_5.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_5.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_opened_one_5.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_opened_one_5.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_opened_one_5.yaml
fi
if [ ${SIX} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_6.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_6.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_6.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_opened_one_6.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_opened_one_6.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_opened_one_6.yaml
fi
if [ ${REALDOUBLE} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_icra_2_lock_1.yaml
fi
fi

if [ ${CE_OBJ} -eq 1 ]
then
if [ ${FOUR} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_4.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_4.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_4.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_opened_one_4.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_opened_one_4.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_opened_one_4.yaml
fi
if [ ${FIVE} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_5.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_5.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_5.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_opened_one_5.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_opened_one_5.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_opened_one_5.yaml
fi
if [ ${SIX} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_6.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_6.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_6.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_opened_one_6.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_opened_one_6.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_opened_one_6.yaml
fi
if [ ${REALDOUBLE} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_icra_2_lock_1.yaml
fi
fi

if [ ${OSCE_OBJ} -eq 1 ]
then
if [ ${FOUR} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o 1s_cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_4.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o 1s_cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_4.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o 1s_cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_4.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o 1s_cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_opened_one_4.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o 1s_cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_opened_one_4.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o 1s_cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_opened_one_4.yaml
fi
if [ ${FIVE} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o 1s_cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_5.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o 1s_cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_5.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o 1s_cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_5.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o 1s_cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_opened_one_5.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o 1s_cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_opened_one_5.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o 1s_cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_opened_one_5.yaml
fi
if [ ${SIX} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o 1s_cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_6.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o 1s_cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_6.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o 1s_cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_6.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o 1s_cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_opened_one_6.yaml
#python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o 1s_cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_opened_one_6.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o 1s_cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_opened_one_6.yaml
fi
if [ ${REALDOUBLE} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o 1s_cross_entropy --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_icra_2_lock_1.yaml
fi
fi


if [ ${HEU_OBJ} -eq 1 ]
then
if [ ${FOUR} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_4.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_4.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_4.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_opened_one_4.yaml
#python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_opened_one_4.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_opened_one_4.yaml
fi
if [ ${FIVE} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_5.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_5.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_5.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_opened_one_5.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_opened_one_5.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_opened_one_5.yaml
fi
if [ ${SIX} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_6.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_6.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_6.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_serial_opened_one_6.yaml
###python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_zigzag_opened_one_6.yaml
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_ngram_opened_one_6.yaml
fi
if [ ${REALDOUBLE} -eq 1 ]
then
python ${JOINT_DEP_EXP_PY_FOLDER}joint_dep_exp.py -o heuristic_proximity --use_simple_locking_state --use_joint_positions --joint_state "small" -r 50 -q 30 --lockboxfile ${LOCKBOXES_YAML_FOLDER}lockbox_icra_2_lock_1.yaml
fi
fi
