import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA

all_results = [
    """
    Ideal 1  1  1  1  1
    CMML	0.7658 	0.8474 	0.5694 	0.6865 	0.8223 
    CMML-NoSA	0.7660 	0.8496 	0.5772 	0.6981 	0.8146 
    CMML-NoCS	0.7115 	0.8066 	0.5137 	0.7846 	0.6593 
    CMML-NoMLP	0.8066 	0.8727 	0.6438 	0.7692 	0.8333 
    LogR	0.6923 	0.7500 	0.4352 	0.4538 	0.8626 
    SVM	0.7596 	0.8244 	0.5110 	0.6615 	0.8297 
    KNN	0.6827 	0.6936 	0.3000 	0.4538 	0.8462 
    DT	0.7019 	0.6904 	0.3809 	0.6215 	0.7593 
    MLP	0.7237 	0.7393 	0.4264 	0.5308 	0.8615 
    Adaboost	0.7564 	0.7972 	0.5044 	0.5846 	0.8791 
    XGBoost	0.7372 	0.8082 	0.5286 	0.5615 	0.8626 
    GBDT	0.7824 	0.8397 	0.6032 	0.6208 	0.8978 
    RF	0.7891 	0.8558 	0.5696 	0.6508 	0.8879 
    GATE	0.6962 	0.7834 	0.4727 	0.6985 	0.6945 
    TabTransformer	0.6631 	0.7150 	0.3708 	0.6231 	0.6918 
    AutoInt	0.6846 	0.8241 	0.5427 	0.3046 	0.9560 
    """,
    """
    Ideal 1  1  1  1  1
    CMML	0.8457 	0.9109 	0.7042 	0.7704 	0.8969 
    CMML-NoSA	0.8401 	0.9140 	0.7097 	0.8282 	0.8481 
    CMML-NoCS	0.8198 	0.8980 	0.6802 	0.7801 	0.8468 
    CMML-NoMLP	0.8326 	0.9116 	0.6907 	0.8206 	0.8408 
    LogR	0.7702 	0.8336 	0.5735 	0.5613 	0.9123 
    SVM	0.7441 	0.8492 	0.5647 	0.4645 	0.9342 
    KNN	0.7076 	0.7772 	0.4726 	0.4129 	0.9079 
    DT	0.7587 	0.7508 	0.5016 	0.7090 	0.7925 
    MLP	0.7321 	0.7903 	0.4625 	0.5981 	0.8232 
    Adaboost	0.7990 	0.8533 	0.5967 	0.6839 	0.8772 
    XGBoost	0.8277 	0.8845 	0.6702 	0.7806 	0.8596 
    GBDT	0.8355 	0.8911 	0.6782 	0.7458 	0.8965 
    RF	0.8535 	0.9133 	0.7130 	0.7768 	0.9057 
    GATE	0.7457 	0.8355 	0.5394 	0.6819 	0.7890 
    TabTransformer	0.7663 	0.8351 	0.5562 	0.7594 	0.7711 
    AutoInt	0.7616 	0.8539 	0.5944 	0.5671 	0.8939 
    """,
    """
    Ideal 1  1  1  1  1
    CMML	0.8589 	0.9287 	0.7381 	0.8103 	0.8882 
    CMML-NoSA	0.8557 	0.9291 	0.7411 	0.8025 	0.8880 
    CMML-NoCS	0.8276 	0.9023 	0.6725 	0.7454 	0.8774 
    CMML-NoMLP	0.8546 	0.9242 	0.7314 	0.8613 	0.8505 
    LogR	0.7333 	0.8002 	0.5014 	0.5470 	0.8462 
    SVM	0.7688 	0.8325 	0.5238 	0.5635 	0.8930 
    KNN	0.7542 	0.8059 	0.5426 	0.5359 	0.8863 
    DT	0.7496 	0.7388 	0.4776 	0.6950 	0.7826 
    MLP	0.7373 	0.7954 	0.4885 	0.6652 	0.7809 
    Adaboost	0.7750 	0.8460 	0.5530 	0.6796 	0.8328 
    XGBoost	0.8188 	0.8871 	0.6359 	0.7017 	0.8896 
    GBDT	0.8004 	0.8712 	0.5792 	0.6961 	0.8635 
    RF	0.8021 	0.8893 	0.6268 	0.7072 	0.8595 
    GATE	0.8029 	0.8425 	0.6115 	0.7691 	0.8234 
    TabTransformer	0.7973 	0.8611 	0.6221 	0.7707 	0.8134 
    AutoInt	0.8185 	0.8811 	0.6446 	0.7564 	0.8562 
    """,
    """
    Ideal 1  1  1  1  1
    CMML	0.8456 	0.9067 	0.7065 	0.8615 	0.8270 
    CMML-NoSA	0.8367 	0.9040 	0.6959 	0.8377 	0.8357 
    CMML-NoCS	0.8294 	0.9029 	0.6958 	0.8510 	0.8041 
    CMML-NoMLP	0.8304 	0.8990 	0.6851 	0.8580 	0.7980 
    LogR	0.6799 	0.7568 	0.4396 	0.7965 	0.5431 
    SVM	0.7780 	0.8388 	0.5497 	0.8442 	0.7005 
    KNN	0.7126 	0.7998 	0.4646 	0.6537 	0.7817 
    DT	0.6850 	0.6824 	0.3647 	0.7160 	0.6487 
    MLP	0.6722 	0.7538 	0.4267 	0.7013 	0.6381 
    Adaboost	0.7664 	0.8550 	0.5802 	0.7792 	0.7513 
    XGBoost	0.7967 	0.8665 	0.6183 	0.7792 	0.8173 
    GBDT	0.7995 	0.8753 	0.6164 	0.8009 	0.7980 
    RF	0.7678 	0.8487 	0.5562 	0.7775 	0.7563 
    GATE	0.8023 	0.8698 	0.6237 	0.8268 	0.7736 
    TabTransformer	0.8147 	0.8782 	0.6517 	0.8407 	0.7843 
    AutoInt	0.8124 	0.8814 	0.6619 	0.8468 	0.7721 
    """,
    """
    Ideal 1  1  1  1  1
    CMML	0.8833 	0.9411 	0.7600 	0.7916 	0.9313 
    CMML-NoSA	0.8862 	0.9451 	0.7722 	0.7966 	0.9330 
    CMML-NoCS	0.8792 	0.9366 	0.7608 	0.8259 	0.9071 
    CMML-NoMLP	0.8844 	0.9410 	0.7694 	0.8329 	0.9113 
    LogR	0.8010 	0.8496 	0.5764 	0.5756 	0.9188 
    SVM	0.8068 	0.8792 	0.6152 	0.5378 	0.9473 
    KNN	0.7888 	0.8015 	0.4903 	0.5315 	0.9232 
    DT	0.7421 	0.7186 	0.4366 	0.6424 	0.7942 
    MLP	0.6901 	0.6835 	0.3812 	0.5935 	0.7405 
    Adaboost	0.7851 	0.7545 	0.5302 	0.6492 	0.8562 
    XGBoost	0.8753 	0.9228 	0.7360 	0.7647 	0.9330 
    GBDT	0.8683 	0.8943 	0.7064 	0.8015 	0.9032 
    RF	0.8723 	0.9317 	0.7275 	0.7987 	0.9108 
    GATE	0.8464 	0.9000 	0.6923 	0.7777 	0.8823 
    TabTransformer	0.8650 	0.9285 	0.7476 	0.8244 	0.8862 
    AutoInt	0.8797 	0.9377 	0.7777 	0.8239 	0.9089 
    """,
    """
    Ideal 1  1  1  1  1
    CMML	0.9300 	0.9664 	0.8664 	0.8909 	0.9473 
    CMML-NoSA	0.9260 	0.9648 	0.8714 	0.8715 	0.9500 
    CMML-NoCS	0.9265 	0.9675 	0.8648 	0.8847 	0.9449 
    CMML-NoMLP	0.9295 	0.9598 	0.8543 	0.8986 	0.9431 
    LogR	0.8153 	0.8369 	0.5662 	0.5859 	0.9163 
    SVM	0.8410 	0.8981 	0.6648 	0.6448 	0.9274 
    KNN	0.8323 	0.8797 	0.6354 	0.6296 	0.9215 
    DT	0.7944 	0.7701 	0.5438 	0.7140 	0.8299 
    MLP	0.7752 	0.7913 	0.5279 	0.6084 	0.8485 
    Adaboost	0.8184 	0.7989 	0.5705 	0.6515 	0.8919 
    XGBoost	0.8807 	0.9278 	0.7164 	0.7458 	0.9400 
    GBDT	0.8548 	0.9145 	0.6944 	0.6923 	0.9264 
    RF	0.8705 	0.9338 	0.7241 	0.7372 	0.9291 
    GATE	0.9253 	0.9597 	0.8453 	0.8754 	0.9472 
    TabTransformer	0.9256 	0.9639 	0.8572 	0.8797 	0.9458 
    AutoInt	0.9269 	0.9655 	0.8691 	0.8697 	0.9520 
    """,
    """
    Ideal 1  1  1  1  1
    CMML	0.9199 	0.9540 	0.8283 	0.8682 	0.9456 
    CMML-NoSA	0.9204 	0.9536 	0.8301 	0.8699 	0.9435 
    CMML-NoCS	0.9197 	0.9632 	0.8299 	0.8817 	0.9372 
    CMML-NoMLP	0.9186 	0.9461 	0.8307 	0.8871 	0.9330 
    LogR	0.7769 	0.8124 	0.4852 	0.4884 	0.9094 
    SVM	0.8089 	0.8645 	0.5933 	0.6379 	0.8875 
    KNN	0.8351 	0.8799 	0.6840 	0.6918 	0.9009 
    DT	0.7781 	0.7454 	0.4895 	0.6492 	0.8374 
    MLP	0.7842 	0.8114 	0.5234 	0.5558 	0.8891 
    Adaboost	0.8021 	0.7750 	0.4955 	0.5855 	0.9016 
    XGBoost	0.8758 	0.9275 	0.7365 	0.7381 	0.9391 
    GBDT	0.8318 	0.8875 	0.6296 	0.6780 	0.9024 
    RF	0.8328 	0.9052 	0.6555 	0.7128 	0.8879 
    GATE	0.9105 	0.9486 	0.8139 	0.8615 	0.9330 
    TabTransformer	0.9180 	0.9598 	0.8319 	0.8786 	0.9361 
    AutoInt	0.9149 	0.9586 	0.8247 	0.8593 	0.9404 
    """,
    """
    Ideal 1  1  1  1  1
    CMML	0.9118 	0.9581 	0.8269 	0.8899 	0.9292 
    CMML-NoSA	0.9064 	0.9617 	0.8198 	0.8822 	0.9257 
    CMML-NoCS	0.9013 	0.9660 	0.8177 	0.8843 	0.9148 
    CMML-NoMLP	0.9006 	0.9533 	0.8143 	0.9048 	0.8973 
    LogR	0.7487 	0.7941 	0.4928 	0.6898 	0.7956 
    SVM	0.8062 	0.8585 	0.6090 	0.7569 	0.8453 
    KNN	0.8215 	0.8810 	0.6322 	0.7685 	0.8637 
    DT	0.7190 	0.7215 	0.4400 	0.6766 	0.7527 
    MLP	0.7378 	0.7845 	0.5037 	0.6796 	0.7842 
    Adaboost	0.7487 	0.8341 	0.5345 	0.7616 	0.7385 
    XGBoost	0.8133 	0.8963 	0.6400 	0.8125 	0.8140 
    GBDT	0.7781 	0.8644 	0.5819 	0.7731 	0.7820 
    RF	0.7878 	0.8838 	0.6025 	0.7588 	0.8109 
    GATE	0.8645 	0.9264 	0.7585 	0.8433 	0.8814 
    TabTransformer	0.8887 	0.9573 	0.7943 	0.8509 	0.9188 
    AutoInt	0.8791 	0.9387 	0.7733 	0.8565 	0.8971 
    """
]

import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE

# Given metrics
metrics = ['ACC', 'AUC', 'KS', 'Type1-ACC', 'Type2-ACC']


# Parse the input data
def parse_results(results):
    models, data = [], []
    for line in results.strip().split('\n'):
        parts = line.split()
        models.append(parts[0])
        data.append([float(x) for x in parts[1:]])
    return models, data


# Assuming 'all_results' is a list of strings
# and each string contains the results for all models on a single dataset.
model_vectors = {model: [] for model in parse_results(all_results[0])[0]}  # Initialize dictionary

for block in all_results:
    models, data = parse_results(block)
    for model, scores in zip(models, data):
        model_vectors[model].extend(scores)

# Now create a list of vectors and a list of model names
vectors = []
model_names = []
for model, vector in model_vectors.items():
    model_names.append(model)
    vectors.append(vector)

df = pd.DataFrame(model_vectors,)
# Perform MDS
mds = MDS(n_components=2, random_state=42)
results = mds.fit_transform(vectors)

# pca = PCA(n_components=2)
# results = pca.fit_transform(vectors)

# tsne = TSNE(n_components=2, random_state=42, perplexity=10)
# results = tsne.fit_transform(np.array(vectors))

# umap_reducer = UMAP(n_components=2)
# results = umap_reducer.fit_transform(vectors)

# Plot the results
plt.figure(figsize=(12, 8))
plt.scatter(results[:, 0], results[:, 1])

for i, txt in enumerate(model_names):
    plt.annotate(txt, (results[i, 0]+.02, results[i, 1]-0.01))

# Find the coordinates of the 'Ideal' model
ideal_index = model_names.index('Ideal')
ideal_coords = results[ideal_index]

# Draw concentric circles around the 'Ideal' model
radii = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2]  # Adjust the radii as needed
for radius in radii:
    circle = plt.Circle(ideal_coords, radius, fill=False, linestyle='--', linewidth=.7)
    plt.gca().add_artist(circle)

plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
# plt.title('MDS Visualization of Model Performance')
plt.grid(False)
plt.show()
