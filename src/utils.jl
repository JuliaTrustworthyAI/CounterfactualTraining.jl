function unwrap(train_set; labels=nothing)
    X, ys = reduce((batch,next_batch) -> (hcat(batch[1],next_batch[1]),hcat(batch[2],next_batch[2])), train_set)
    ycold = (x -> reduce(vcat, x))([findall(y) for y in eachcol(ys)])
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
    return acc / (train_set.batchsize * length(train_set))
end