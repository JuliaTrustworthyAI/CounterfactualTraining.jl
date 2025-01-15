using CTExperiments
using DotEnv

# Setup:
DotEnv.load!()

# Get config and set up grid:
config_file = get_config_from_args(; save_adjusted=false, return_adjusted=false)
root_name = CTExperiments.from_toml(config_file)["name"]
exper_grid = ExperimentGrid(config_file)

# Determine number of slurm tasks:
total_tasks = ntasks(exper_grid)
_nthreads = parse(Int, ENV["NTHREADS"])
while total_tasks * _nthreads > parse(Int, ENV["MAX_TASKS"])
    global total_tasks /= 2
end
total_tasks = round(Int, total_tasks)

@info "Requesting $(total_tasks * _nthreads) resources for experiment '$root_name'. CPUs: $total_tasks; Threads per task: $(_nthreads)."

println(total_tasks)
