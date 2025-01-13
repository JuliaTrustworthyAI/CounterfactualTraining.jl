using CTExperiments
using CTExperiments.CounterfactualExplanations
using Serialization

for (k,v) in CTExperiments.data_sets
    if k == "mnist"
        vae = CounterfactualExplanations.Models.load_mnist_vae()
    end
    ce_data = get_ce_data(v(); train_only=true)
    nin = CTExperiments.input_dim(v())
    if nin <= 2
        vae = CounterfactualExplanations.DataPreprocessing.fit_transformer(
            ce_data, CounterfactualExplanations.GenerativeModels.VAE;
            gpu=false,
            latent_dim=2,
            hidden_dim=16, 
        )
    else
        vae = CounterfactualExplanations.DataPreprocessing.fit_transformer(
            ce_data, CounterfactualExplanations.GenerativeModels.VAE; gpu=false,
            latent_dim=4,
            hidden_dim=32, 
        )
    end
    Serialization.serialize(joinpath(mkpath("paper/experiments/dev/vae"), "$k.jls"), vae)
end