import torch, os
from config import cfg
from models import init_models, init_models_faulty
import zs_hooks_stats as stats
from autoattack import AutoAttack


def attacking(
    testloader,
    arch,
    dataset,
    in_channels,
    precision,
    checkpoint_path,
    device,
    faulty_layers,
    ber,
    position,
    norm,
    epsilon,
):

    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    if ber or "eopm" in checkpoint_path or "adversarial" in checkpoint_path:
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
    else:
        model, __ = init_models(
            arch,
            in_channels,
            precision,
            True,
            checkpoint_path,
            dataset=dataset,
        )

    model.to(device)
    model.eval()

    adversary = AutoAttack(
        model,
        norm=norm,
        eps=epsilon,
        version="standard",
        device=device,
    )

    images, labels = next(iter(testloader))

    # adversary.attacks_to_run = ['apgd-ce', 'fab']
    # adversary.apgd.n_restarts = 2
    # adversary.fab.n_restarts = 2
    adversary.square.n_restarts = 10
    adversary.square.n_queries = 15000

    with torch.no_grad():
        print(
            "Run standard Auto-Attack evaluation with",
            norm,
            "norm and epsilon",
            epsilon,
        )
        # print(images.shape)
        # print(labels.shape)
        adv_complete = adversary.run_standard_evaluation(
            images,
            labels,
            bs=testloader.batch_size,
        )

        torch.save(
            {"adv_complete": adv_complete},
            "{}/{}_{}_1_{}_eps_{:.5f}_norm_{}.pth".format(
                cfg.save_dir,
                "aa",
                "standard",
                adv_complete.shape[0],
                epsilon,
                norm,
            ),
        )
