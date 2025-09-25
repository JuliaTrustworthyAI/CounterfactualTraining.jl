## Training



## Evaluation

sbatch --time=01:30:00 --ntasks=40 --mem-per-cpu=3800MB $RUN_EVAL_SEQ --inherit=false --model=mlp --test_time=true --n_runs=100 --n_individuals=250 --data=lin_sep

sbatch --time=01:30:00 --ntasks=40 --mem-per-cpu=3800MB $RUN_EVAL_SEQ --inherit=false --model=mlp --test_time=true --n_runs=100 --n_individuals=250 --data=circles

sbatch --time=01:30:00 --ntasks=40 --mem-per-cpu=3800MB $RUN_EVAL_SEQ --inherit=false --model=mlp --test_time=true --n_runs=100 --n_individuals=250 --data=moons

sbatch --time=01:30:00 --ntasks=40 --mem-per-cpu=3800MB $RUN_EVAL_SEQ --inherit=false --model=mlp --test_time=true --n_runs=100 --n_individuals=250 --data=over

sbatch --time=01:00:00 --ntasks=40 --mem-per-cpu=3800MB $RUN_EVAL_SEQ --inherit=false --model=mlp --test_time=true --n_runs=100 --n_individuals=100 --data=gmsc

sbatch --time=01:00:00 --ntasks=40 --mem-per-cpu=3800MB $RUN_EVAL_SEQ --inherit=false --model=mlp --test_time=true --n_runs=100 --n_individuals=100 --data=cali

sbatch --partition=memory --time=01:00:00 --ntasks=20 --mem-per-cpu=16GB $RUN_EVAL_SEQ --inherit=false --model=mlp --test_time=true --n_runs=100 --n_individuals=100 --data=adult

sbatch --partition=memory --time=01:00:00 --ntasks=20 --mem-per-cpu=16GB $RUN_EVAL_SEQ --inherit=false --model=mlp --test_time=true --n_runs=100 --n_individuals=100 --data=credit
