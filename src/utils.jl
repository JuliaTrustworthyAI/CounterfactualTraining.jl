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

    # Decode one-hot labels:
    ycold = (x -> reduce(vcat, x))([findall(y) for y in eachcol(ys)])

    # If labels are provided, replace indices with labels:
    if !isnothing(labels)
        @assert length(labels) == size(ys, 1)
        replace!(ycold, [i => label for (i, label) in enumerate(labels)]...)
    end
    return X, ycold
end

function accuracy(model, train_set)
    acc = 0
    for (x, y) in train_set
        yhat = [argmax(_x) for _x in eachcol(softmax(model(x)))]
        y = Flux.onecold(y)
        acc += sum(yhat .== y)
    end
    return acc / size(train_set.data[1],2)
end

"""
    infer_domain_constraints(X::AbstractArray; nstd=3)

Automatically infers reasonable domain constraints for the counterfactuals. 
"""
function infer_domain_constraints(X::AbstractArray; nstd=3)
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
