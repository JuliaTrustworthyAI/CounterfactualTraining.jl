using CTExperiments
using DotEnv

# Setup:
DotEnv.load!()

# Get config and set up grid:
config_file = get_config_from_args(; save_adjusted=false, return_adjusted=true)
@info "After `get_config_from_args` call ..."
root_name = CTExperiments.from_toml(config_file)["name"]
exper_grid = ExperimentGrid(config_file)
@info "After setting up grid..."

# Determine number of slurm tasks:
total_tasks = ntasks(exper_grid)
@assert total_tasks > 0 "It seems that all tasks have already been completed."
_nthreads = parse(Int, ENV["NTHREADS"])
while total_tasks * _nthreads > parse(Int, ENV["MAX_TASKS"])
    global total_tasks /= 2
end
total_tasks = round(Int, total_tasks)

@info "Requesting $(total_tasks * _nthreads) resources for experiment '$root_name'. CPUs: $total_tasks; Threads per task: $(_nthreads)."

println(total_tasks)
