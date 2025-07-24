using TaijaData
using JLD2
using IntegratedGradients
using Plots
using Plots.PlotMeasures
using Statistics

function test_MNIST(model1_path, model2_path; steps=50, baseline_type="zeros")
    data = TaijaData.load_mnist_test()
    xs_s, ys_s = data[1], data[2]
    model1 = JLD2.load(model1_path, "model")
    model2 = JLD2.load(model2_path, "model")

    for i in 0:9
        println("Class $i")
        indices = findall(==(i), ys_s)
        xs = xs_s[:, indices]
        ys = ys_s[indices]
        protected_cols = [1, 2, 3, 4, 5, 24, 25, 26, 27, 28]

        # Get average contributions for Model 1
        IG1 = calculate_average_contributions(
            model1, xs, ys; steps=steps, baseline_type="random"
        )
        # Normalize into 0-1 range
        zero_IG1 = round((0 - minimum(IG1)) / (maximum(IG1) - minimum(IG1)); sigdigits=4)
        IG1 = (IG1 .- minimum(IG1)) ./ (maximum(IG1) - minimum(IG1))
        # Reshape into matrix
        IG1 = reshape(IG1, 28, 28)

        # Calculate mean only over protected columns
        mean_IG1 = round(mean(IG1[:, protected_cols]); sigdigits=4)
        # Calculate standard error only over protected columns
        std_err_IG1 = round(
            std(IG1[:, protected_cols]) ./ sqrt(length(indices)); sigdigits=4
        )
        # Print results
        println("$mean_IG1 +/- $std_err_IG1")

        # Save a heatmap, rotated 90° left to obtain proper orientation
        savefig(
            heatmap(
                mapslices(rotl90, IG1; dims=[1, 2]);
                aspect_ratio=:equal,
                seriescolor=:thermal,
                axis=false,
                ticks=nothing,
                border=false,
                colorbar=false,
                framestyle=:none,
                margin=0px,
                padding=0px,
                size=(600, 600),
            ),
            "data/mnist1_$i.png",
        )

        # -------------------------

        # Get average contributions for Model 2
        IG2 = calculate_average_contributions(
            model2, xs, ys; steps=steps, baseline_type="random"
        )
        # Normalize into 0-1 range
        zero_IG2 = round((0 - minimum(IG2)) / (maximum(IG2) - minimum(IG2)); sigdigits=4)
        IG2 = (IG2 .- minimum(IG2)) ./ (maximum(IG2) - minimum(IG2))
        # Reshape into matrix
        IG2 = reshape(IG2, 28, 28)

        # Calculate mean only over protected columns
        mean_IG2 = round(mean(IG2[:, protected_cols]); sigdigits=4)
        # Calculate standard error only over protected columns
        std_err_IG2 = round(
            std(IG2[:, protected_cols]) ./ sqrt(length(indices)); sigdigits=4
        )
        # Print results
        println("$mean_IG2 +/- $std_err_IG2")

        # Save a heatmap, rotated 90° left to obtain proper orientation
        savefig(
            heatmap(
                mapslices(rotl90, IG2; dims=[1, 2]);
                aspect_ratio=:equal,
                seriescolor=:thermal,
                axis=false,
                ticks=nothing,
                border=false,
                colorbar=false,
                framestyle=:none,
                margin=0px,
                padding=0px,
                size=(600, 600),
            ),
            "data/mnist2_$i.png",
        )

        println()
        println("Zero point CT: $zero_IG1")
        println("Zero point vanilla: $zero_IG2")
        println()
    end
end
