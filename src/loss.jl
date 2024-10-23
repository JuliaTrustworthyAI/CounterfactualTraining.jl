using EnergySamplers: EnergySamplers

"""
    contrastive_loss(yhat1, yhat2, y; ϵ=0.5)

Computes the contrastive loss as in Chopra et al. ([2005](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf)).
"""
function contrastive_loss(yhat1, yhat2, y; ϵ=0.5)
    # Compute the Euclidean distance between the two embeddings
    D = norm(yhat1 - yhat2)

    # Loss for positive pairs (same class, y = 1)
    positive_loss = y * (D^2)

    # Loss for negative pairs (different class, y = 0)
    negative_loss = (1 - y) * max(0, ϵ - D)^2

    # Total loss is the sum of the two components
    return positive_loss + negative_loss
end

