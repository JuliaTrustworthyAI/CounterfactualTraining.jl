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

    # If labels are provided, map indices to labels (builds a new vector
    # with the correct element type — replace! cannot change eltype):
    if !isnothing(labels)
        @assert length(labels) == size(ys, 1)
        ycold = [labels[i] for i in ycold]
    end
    return X, ycold
end

"""
    accuracy(model, train_set; device=identity)

Compute classification accuracy over a `DataLoader`. Evaluates the model in
test (eval) mode so BatchNorm uses running rather than batch statistics, and
accumulates match counts on the device, syncing to the host once per call.
The `device` keyword moves each batch to the device before the forward pass.
"""
function accuracy(model, train_set; device=identity)
    acc_dev = device([0])
    Flux.testmode!(model)
    try
        for (x, y) in train_set
            x = x |> device
            y = y |> device
            logits = model(x)
            # argmax(logits) == argmax(softmax(logits)) — softmax is monotone per
            # column, so it can be skipped without changing predictions.
            yhat = vec(argmax(logits; dims=1))
            y_true = vec(argmax(y; dims=1))
            acc_dev .+= sum(yhat .== y_true; dims=1)   # dims=1 keeps result on device (no sync)
        end
    finally
        Flux.trainmode!(model)
    end
    return Flux.cpu(acc_dev)[1] / size(train_set.data[1], 2)
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
