using StatsBase

"""
    unwrap(train_set; labels=nothing)

Unwraps the data from a Flux.DataLoader or zip iterator. The output variables is assumed to be categorical. If no `labels` are provided, then 1 to n is used where n is the number of classes.
"""
function unwrap(train_set; labels=nothing)

    # Concatenate batches:
    X, ys = reduce(
        (batch, next_batch) ->
            (hcat(batch[1], next_batch[1]), hcat(batch[2], next_batch[2])),
        train_set,
    )

    # Move labels to CPU before decoding (findall uses scalar indexing,
    # which is disallowed on GPU arrays)
    ys = Flux.cpu(ys)

    # Decode one-hot labels:
    ycold = (x -> reduce(vcat, x))([findall(y) for y in eachcol(ys)])

    # If labels are provided, replace indices with labels:
    if !isnothing(labels)
        @assert length(labels) == size(ys, 1)
        replace!(ycold, [i => label for (i, label) in enumerate(labels)]...)
    end
    return X, ycold
end

"""
    accuracy(model, train_set; device=identity)

Compute classification accuracy over a `DataLoader`. Uses `Flux.onecold` on
whole matrices — GPU-compatible and faster than per-column `argmax`.
The `device` keyword moves each batch to the device before the forward pass.
"""
function accuracy(model, train_set; device=identity)
    acc = 0
    for (x, y) in train_set
        x = x |> device
        # Move logits to CPU before onecold (mapslices/argmax may scalar-index GPU arrays)
        logits = model(x) |> Flux.cpu
        yhat = Flux.onecold(Flux.softmax(logits))
        y_true = Flux.onecold(y)
        acc += sum(yhat .== y_true)
    end
    return acc / size(train_set.data[1], 2)
end

"""
    infer_domain_constraints(X::AbstractArray; nstd=3)

Automatically infers reasonable domain constraints for the counterfactuals. 
"""
function infer_domain_constraints(X::AbstractArray; nstd=2)
    bounds = Tuple[]
    for x in eachrow(X)
        xmin, xmax = extrema(x)
        sigma = std(x)
        mu = mean(x)
        lb, ub = (mu - nstd * sigma, mu + nstd * sigma)
        push!(bounds, (minimum([lb, xmin]), maximum([ub, xmax])))
    end
    return bounds
end
