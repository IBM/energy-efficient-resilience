import torch
import os
from config import cfg
from models import init_models, init_models_faulty
import zs_hooks_stats as stats
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from torch.utils.data import Dataset, DataLoader


class AdversarialImageSet(Dataset):
    def __init__(self, img_loc):
        self.img_loc = img_loc
        self.images = torch.load(img_loc)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        return image


# adapted from: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
def pgd_attack(model, image, target, epsilon=0.3, alpha=5 / 255, iters=10):
    perturbed_image = torch.clone(image) + torch.FloatTensor(
        image.shape
    ).uniform_(-epsilon, epsilon)
    cross_entropy = torch.nn.CrossEntropyLoss()
    for i in range(iters):
        model_outputs = model(image)
        loss = cross_entropy(model_outputs, target)
        model.zero_grad(set_to_none=True)
        loss.backward()
        perturbation = torch.clamp(
            alpha * image.grad.data.sign(), -epsilon, epsilon
        )
        perturbed_image = perturbation + image
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    return perturbed_image


def attacking(
    test_loader,
    arch,
    dataset,
    in_channels,
    precision,
    checkpoint_path,
    device,
    faulty_layers,
    ber,
    position,
    epsilon,
):

    random.seed(2022)

    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    fig_folder = (
        "./output_ber_" + "{:.2f}".format(ber) + "_eps_" + str(epsilon)
    )
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    if ber != 0:
        try:
            model, __ = init_models_faulty(
                arch,
                in_channels,
                precision,
                True,
                checkpoint_path,
                faulty_layers,
                ber,
                position,
                dataset=dataset,
            )

            stats.resnet_register_hooks(model, arch)
        except KeyError:
            print("Use of generators not supported yet")
    else:
        model, __ = init_models(
            arch,
            in_channels,
            precision,
            True,
            checkpoint_path,
            dataset=dataset,
        )

        stats.resnet_register_hooks(model, arch)

    model.eval()
    model.to(device)

    print("Run PGD adversarial attack evaluation with epsilon", epsilon)

    running_correct_std = 0
    running_correct_adv = 0

    running_preds_std = torch.zeros(0)
    running_preds_adv = torch.zeros(0)

    running_logits_std = np.empty((1, 10))
    running_logits_adv = np.empty((1, 10))

    images_std = torch.zeros(0)
    images_adv = torch.zeros(0)
    labels = torch.zeros(0)

    perturb = True

    adv_tensor_loc = 'cifar10_adv_images_ber_' + \
        "{:.2f}".format(ber) + "_eps_" + str(epsilon) + '.pth'
    if os.path.exists(adv_tensor_loc):
        adv_dataset = AdversarialImageSet(adv_tensor_loc)
        adv_loader = DataLoader(adv_dataset, batch_size=cfg.test_batch_size)
        perturb = False
        iter_adv = iter(adv_loader)

    for t, (inputs, classes) in enumerate(test_loader):
        __, (ax1, ax2) = plt.subplots(1, 2)
        index = random.randint(0, cfg.test_batch_size - 1)

        inputs = inputs.to(device)
        classes = classes.to(device)

        inputs.requires_grad = True

        model_outputs = model(inputs)
        running_logits_std = np.vstack(
            (running_logits_std, model_outputs.detach().numpy())
        )
        __, preds = torch.max(model_outputs, 1)
        running_preds_std = torch.cat((running_preds_std, preds))
        correct = torch.sum(preds == classes.data)
        running_correct_std += correct

        trans_img = torch.transpose(torch.transpose(inputs[index], 0, 2), 0, 1)
        adjust_img = (trans_img + 1) / 2
        ax1.imshow((adjust_img * 255).int())
        ax1.set_title(
            "Normal\nPredicted: "
            + str(int(preds[index]))
            + ", Actual: "
            + str(int(classes[index]))
        )

        if perturb:
            perturbed_data = pgd_attack(
                model, inputs, classes, epsilon=epsilon)
        else:
            perturbed_data = next(iter_adv)

        images_std = torch.cat((images_std, inputs))
        labels = torch.cat((labels, classes))

        images_adv = torch.cat((images_adv, perturbed_data))

        perturbed_output = model(perturbed_data)
        running_logits_adv = np.vstack(
            (running_logits_adv, perturbed_output.detach().numpy())
        )
        __, preds = torch.max(perturbed_output, 1)
        running_preds_adv = torch.cat((running_preds_adv, preds))
        correct = torch.sum(preds == classes.data)
        running_correct_adv += correct

        trans_img = torch.transpose(
            torch.transpose(perturbed_data[index], 0, 2), 0, 1
        )
        adjust_img = (trans_img + 1) / 2
        ax2.imshow((adjust_img * 255).int())
        ax2.set_title(
            "Adversary\nPredicted: "
            + str(int(preds[index]))
            + ", Actual: "
            + str(int(classes[index]))
        )
        plt.savefig(fig_folder + "/img_" + str(int(t)) + ".png")
        plt.close("all")

        print(
            int(t) + 1,
            "/",
            (int(len(test_loader.dataset) / cfg.test_batch_size)),
            "batches",
            end="\r",
        )

    if perturb:
        torch.save(images_adv, adv_tensor_loc)

    print(
        "\nStandard Eval Accuracy %.3f"
        % (running_correct_std.double() / (len(test_loader.dataset)))
    )

    print(
        "Adversary Eval Accuracy %.3f"
        % (running_correct_adv.double() / (len(test_loader.dataset)))
    )

    # ---------------------------- TSNE PLOTS ----------------------------#
    running_logits_std = np.delete(running_logits_std, 0, axis=0)
    running_logits_adv = np.delete(running_logits_adv, 0, axis=0)

    tsne_data_std = np.zeros((running_logits_std.shape[0], 2))
    tsne_data_adv = np.zeros((running_logits_adv.shape[0], 2))

    for i in range(10):
        tsne_data_std = tsne_data_std + TSNE(
            n_components=2, verbose=0, perplexity=40, n_iter=300
        ).fit_transform(running_logits_std)
        tsne_data_adv = tsne_data_adv + TSNE(
            n_components=2, verbose=0, perplexity=40, n_iter=300
        ).fit_transform(running_logits_adv)
        print(
            i + 1,
            "/",
            10,
            "TSNE runs",
            end="\r",
        )

    tsne_data_std = tsne_data_std / 10
    tsne_data_adv = tsne_data_adv / 10

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    fig, axes = plt.subplots(1, 2, figsize=(18, 18))
    axes.flat[0].set_title("Normal Logits")
    # axes.flat[0].set_xlim([-6, 4])
    # axes.flat[0].set_ylim([-3, 3])
    axes.flat[1].set_title("Adversarial Logits")
    # axes.flat[1].set_xlim([-6, 4])
    # axes.flat[1].set_ylim([-3, 3])
    for i in range(labels.shape[0]):
        axes.flat[0].scatter(
            tsne_data_std[i, 0], tsne_data_std[i, 1], c=colors[int(labels[i])]
        )
        axes.flat[1].scatter(
            tsne_data_adv[i, 0], tsne_data_adv[i, 1], c=colors[int(labels[i])]
        )
    plt.savefig(fig_folder + "/tsne_scatter_logit.png")
    plt.close("all")

    reformat_images_std = (
        torch.flatten(images_std, start_dim=1).detach().numpy()
    )

    reformat_images_adv = (
        torch.flatten(images_adv, start_dim=1).detach().numpy()
    )

    pca_data_std = PCA(n_components=50, svd_solver="full").fit_transform(
        reformat_images_std
    )

    pca_data_adv = PCA(n_components=50, svd_solver="full").fit_transform(
        reformat_images_adv
    )

    tsne_data_std = torch.zeros((reformat_images_std.shape[0], 2))
    tsne_data_adv = torch.zeros((reformat_images_adv.shape[0], 2))

    for i in range(10):
        tsne_data_std = tsne_data_std + TSNE(
            n_components=2, verbose=0, perplexity=40, n_iter=300
        ).fit_transform(pca_data_std)
        tsne_data_adv = tsne_data_adv + TSNE(
            n_components=2, verbose=0, perplexity=40, n_iter=300
        ).fit_transform(pca_data_adv)
        print(
            i + 1,
            "/",
            10,
            "TSNE runs",
            end="\r",
        )

    tsne_data_std = tsne_data_std / 10
    tsne_data_adv = tsne_data_adv / 10

    fig, axes = plt.subplots(2, 5, figsize=(30, 18))
    for i in range(10):
        # axes.flat[i].set_xlim([-11, 11])
        # axes.flat[i].set_ylim([-11, 11])
        axes.flat[i].set_title("Class %i\nDistribution" % i)
    for i in range(labels.shape[0]):
        axes.flat[int(labels[i])].scatter(
            tsne_data_std[i, 0], tsne_data_std[i, 1], c="deepskyblue"
        )
        axes.flat[int(labels[i])].scatter(
            tsne_data_adv[i, 0], tsne_data_adv[i, 1], c="hotpink"
        )
    plt.savefig(fig_folder + "/tsne_scatter_img.png")
    plt.close("all")


# ---------- EXPERIMENTAL ---------- #
# running_logits_std = np.delete(running_logits_std, 0, axis=0)
# running_logits_adv = np.delete(running_logits_adv, 0, axis=0)

# tsne_data_std = np.zeros((running_logits_std.shape[0], 2))
# tsne_data_adv = np.zeros((running_logits_adv.shape[0], 2))

# for i in range(10):
#     tsne_data_std = tsne_data_std + TSNE(
#         n_components=2, verbose=0, perplexity=40, n_iter=300
#     ).fit_transform(running_logits_std)
#     tsne_data_adv = tsne_data_adv + TSNE(
#         n_components=2, verbose=0, perplexity=40, n_iter=300
#     ).fit_transform(running_logits_adv)

# tsne_data_std = tsne_data_std / 10
# tsne_data_adv = tsne_data_adv / 10

# fig, axes = plt.subplots(2, 5, figsize=(30, 18))
# for i in range(10):
#     # axes.flat[i].set_xlim([-11, 11])
#     # axes.flat[i].set_ylim([-11, 11])
#     axes.flat[i].set_title("Class %i\nDistribution" % i)
# for i in range(labels.shape[0]):
#     axes.flat[int(labels[i])].scatter(
#         tsne_data_std[i, 0], tsne_data_std[i, 1], c="deepskyblue"
#     )
#     axes.flat[int(labels[i])].scatter(
#         tsne_data_adv[i, 0], tsne_data_adv[i, 1], c="hotpink"
#     )
# plt.savefig(fig_folder + "/tsne_scatter_logit.png")
# plt.close("all")

# geometric separability
# gsi = geometrical_separability_index(
#     reformat_images_std, labels.cpu().detach().numpy()
# )
# print("Thornton's Geometric Separability for Normal Images", gsi)

# gsi = geometrical_separability_index(
#     reformat_images_adv, labels.cpu().detach().numpy()
# )
# print("Thornton's Geometric Separability for Adversarial Images", gsi)


# cos_sim = cos(running_logits_std, running_logits_adv)
# print(
#     "Cosine Similarity Between Standard and Adversary Logits:"
# )
# for val in cos_sim:
#     print(str(float(val)))

# # separability index
# gsi_std = sep.calculate_separability(reformat_images_std, std_labels, distfun = dis.cosine, show_graph=False)
# print(
#     'Multiscale\'s Separability Index for Normal Images\n', gsi_std[0]['multiscale_separability']
# )

# gsi_adv = sep.calculate_separability(reformat_images_adv, adv_labels, distfun = dis.cosine, show_graph=False)
# print(
#     'Multiscale\'s Separability Index for Adversarial Images\n', gsi_adv[0]['multiscale_separability']
# )

# plt.plot(gsi_std[0]['distance'], gsi_std[0]['multiscale_separability'], color = 'deepskyblue')
# plt.plot(gsi_adv[0]['distance'], gsi_adv[0]['multiscale_separability'], color = 'hotpink')
# plt.xlabel('Search Radius')
# plt.ylabel('Multiscale Separability')
# plt.xlim(0.0, 1.01)
# plt.ylim(0.0, 1.01)
# plt.savefig(fig_folder + '/separability_index.png')
# plt.close('all')

# separating hyperplane
# select smaller subset of images to tra2in on
# indices = torch.randperm(tsne_data_std.shape[0])[:500]
# tsne_data_std = tsne_data_std[indices]
# tsne_data_adv = tsne_data_adv[indices]
# std_labels = std_labels[indices]

# # C_RANGE = np.logspace(-3, 3, 5)
# # param_grid = { 'C': [1],
# #             #    'gamma': ['scale', 'auto'],
# #                'degree': [3],
# #                'kernel': ['poly'] }

# # svm_std = GridSearchCV(SVC(), param_grid, n_jobs = -1, verbose = 0)
# svm_std = SVC(kernel='poly', degree=3, C=1)
# svm_std.fit(tsne_data_std, std_labels) ## fit classifier to data

# # Copt = svm_std.best_params_['C'] ## svm cost parameter
# # Kopt = svm_std.best_params_['kernel'] ## kernel function
# # Gopt = svm_std.best_params_['gamma'] ## gamma of RBF kernel
# # Dopt = svm_std.best_params_['degree'] ## degree of polynomial kernel

# # print('\nOptimal standard SVM parameter values:')
# # print('C:', Copt)
# # print('kernel:', Kopt)
# # print('gamma:', Gopt)
# # print('degree:', Dopt, '\n')

# print('Calculating metrics...') ## generate report
# scores = cross_val_score(svm_std, tsne_data_std, std_labels, cv = 6)
# print('\nAverage cross-validate score: ', scores.mean())

# plt.scatter(tsne_data_std[:, 0], tsne_data_std[:, 1], c=std_labels, s=30, cmap=plt.cm.Paired)

# # plot the decision function
# ax = plt.gca()
# # DecisionBoundaryDisplay.from_estimator(
# #     svm_std,
# #     tsne_data_std,
# #     plot_method="contour",
# #     colors="k",
# #     levels=[-1, 0, 1],
# #     alpha=0.5,
# #     linestyles=["--", "-", "--"],
# #     ax=ax,
# # )
# # plot support vectors
# ax.scatter(
#     svm_std.support_vectors_[:, 0],
#     svm_std.support_vectors_[:, 1],
#     s=100,
#     linewidth=1,
#     facecolors="none",
#     edgecolors="k",
# )
# plt.savefig(fig_folder + '/std_support_vecs.png')
# plt.close('all')

# # svm_adv = GridSearchCV(SVC(), param_grid, n_jobs = -1, verbose = 0)
# svm_adv = SVC(kernel='poly', degree=3, C=1)
# svm_adv.fit(tsne_data_adv, std_labels) ## fit classifier to data

# # Copt = svm_adv.best_params_['C'] ## svm cost parameter
# # Kopt = svm_adv.best_params_['kernel'] ## kernel function
# # Gopt = svm_adv.best_params_['gamma'] ## gamma of RBF kernel
# # Dopt = svm_adv.best_params_['degree'] ## degree of polynomial kernel

# # print('\nOptimal adversarial SVM parameter values:')
# # print('C:', Copt)
# # print('kernel:', Kopt)
# # print('gamma:', Gopt)
# # print('degree:', Dopt, '\n')

# print('Calculating metrics...') ## generate report
# scores = cross_val_score(svm_adv, tsne_data_adv, std_labels, cv = 6)
# print('\nAverage cross-validate score: ', scores.mean())

# plt.scatter(tsne_data_adv[:, 0], tsne_data_adv[:, 1], c=std_labels, s=30, cmap=plt.cm.Paired)

# # plot the decision function
# ax = plt.gca()
# # DecisionBoundaryDisplay.from_estimator(
# #     svm_adv,
# #     tsne_data_adv,
# #     plot_method="contour",
# #     colors="k",
# #     levels=[-1, 0, 1],
# #     alpha=0.5,
# #     linestyles=["--", "-", "--"],
# #     ax=ax,
# # )
# # plot support vectors
# ax.scatter(
#     svm_adv.support_vectors_[:, 0],
#     svm_adv.support_vectors_[:, 1],
#     s=100,
#     linewidth=1,
#     facecolors="none",
#     edgecolors="k",
# )
# plt.savefig(fig_folder + '/adv_support_vecs.png')
# plt.close('all')

# # https://github.com/aacevedot/gsindex/blob/master/src/gsindex.py
# def geometrical_separability_index(matrix, labels):
#     size = len(labels)
#     distances = cdist(matrix, matrix)
#     positions = distances.argsort(axis=0)
#     elements = np.take(labels, positions)
#     reference_points = elements[0]
#     nearest_neighbours = elements[1]
#     gsi = np.sum(reference_points == nearest_neighbours) / size
#     return gsi

# # cosine similarity
# cos = torch.nn.CosineSimilarity(dim=0)
# cos_sim = cos(running_preds_std, running_preds_adv)
# print(
#     "Cosine Similarity Between Standard and Adversary Predictions %.3f"
#     % (cos_sim)
# )
