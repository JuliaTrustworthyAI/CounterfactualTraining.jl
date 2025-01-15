using CTExperiments
using DotEnv

# Setup:
DotEnv.load!()

# Get config and set up grid:
config_file = get_config_from_args()
root_name = CTExperiments.from_toml(config_file)["name"]
exper_grid = ExperimentGrid(config_file)

# Determine number of slurm tasks:
total_tasks = ntasks(exper_grid)
while total_tasks * parse(Int, ENV["NTHREADS"]) > parse(Int, ENV["MAX_TASKS"])
    global total_tasks /= 2
end
total_tasks = round(Int, total_tasks)

@info "Requesting $total_tasks CPUs for experiment '$root_name'. Threads per task: $(parse(Int, ENV["NTHREADS"]))"

println(total_tasks)
