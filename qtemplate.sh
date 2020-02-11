# Keras job submission script
#
# module depends on hamilton 
#
#$ -V
#$ -S /bin/bash
#$ -cwd
#$ -P em
#$ -l gpu=1
#$ -N keras_ham
#$ -l m_mem_free=12G
#$ -l exclusive
#$ -o nn_job.out -e nn_job.err
#$ -q all.q

"$@"

