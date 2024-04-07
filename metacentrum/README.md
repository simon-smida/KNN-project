This directory contains scripts for executing workloads on Metacentrum. The documentation can be found
[at this URL](https://docs.metacentrum.cz/).

Basically, once you have created a user account, execute the job

```shell
# Log into the closest frontend node
ssh <USERNAME>@zenith.metacentrum.cz  
cd /storage/brno2/home/<USERNAME>

git clone https://github.com/simon-smida/KNN-project knn
cd knn && git switch <branch> && cd ..
bash knn/metacentrum/evaluate meta "speechbrain/spkrec-ecapa-voxceleb" "2:00:00"
```

To inspect the job, run

```shell
# This command will show you the job status, the expected launch time, ...
qstat -f <job-ID>
```

To view the logs

```shell
vi batch_job_knn.o<job-ID>
```

To copy results from the cluster to your local machine, run

```shell
scp <USERNAME>@skirit.ics.muni.cz:/storage/brno2/home/<USERNAME>/results/evaluate-<DATE>/det_curve.png
```

To debug in interactive mode, run

```shell
qsub -I -l walltime="2:00:00" -q gpu -l select=1:ncpus=2:ngpus=1:gpu_mem=20gb:mem=20gb:scratch_local=30gb
```
