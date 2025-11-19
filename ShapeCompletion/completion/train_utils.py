import torch
from matplotlib import pyplot as plt
import os
import logging


class AverageValueMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save_model(path, net, net_d=None):
    if net_d is not None:
        torch.save({'net_state_dict': net.module.state_dict(),
                    'D_state_dict': net_d.module.state_dict()}, path)
    else:
        torch.save({'net_state_dict': net.module.state_dict()}, path)


def generator_step(net_d, out2, net_loss, optimizer):
    set_requires_grad(net_d, False)
    d_fake = net_d(out2[:, 0:2048, :])
    errG_loss_batch = torch.mean((d_fake - 1) ** 2)
    total_gen_loss_batch = errG_loss_batch + net_loss * 200
    total_gen_loss_batch.backward(torch.ones(1).to('cuda'), retain_graph=True, )
    optimizer.step()
    return d_fake


def discriminator_step(net_d, gt, d_fake, optimizer_d):
    set_requires_grad(net_d, True)
    d_real = net_d(gt[:, 0:2048, :])
    d_loss_fake = torch.mean(d_fake ** 2)
    d_loss_real = torch.mean((d_real - 1) ** 2)
    errD_loss_batch = 0.5 * (d_loss_real + d_loss_fake)
    total_dis_loss_batch = errD_loss_batch
    total_dis_loss_batch.backward(torch.ones(1).to('cuda'))
    optimizer_d.step()



def plot_pcd_one_view(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3 * 1.4, 3 * 1.4))
    elev = 30
    azim = -45
    for j, (pcd, size) in enumerate(zip(pcds, sizes)):
        color = pcd[:, 0]
        ax = fig.add_subplot(1, len(pcds), j + 1, projection='3d')
        ax.view_init(elev, azim)
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1.0, vmax=0.5)
        ax.set_title(titles[j])
        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)


def setup_wandb(args):
    """
    Initialize W&B based on args.wandb from YAML.

    Expected YAML structure:
      wandb:
        enabled: bool
        api_key: "" or "<KEY>"
        project: "" or "<PROJECT_NAME>"
        tags: "" or ["tag1", "tag2"] or "single_tag"

    Returns:
      (wandb_enabled: bool, wandb_run: wandb.Run or None)
    """
    # try to import wandb locally inside the function
    try:
        import wandb
    except ImportError:
        logging.info("wandb package not installed. W&B disabled.")
        return False, None

    wandb_cfg = getattr(args, "wandb", None)
    if wandb_cfg is None:
        logging.info("No 'wandb' block in config. W&B disabled.")
        return False, None

    # enabled flag from YAML
    if not getattr(wandb_cfg, "enabled", False):
        logging.info("W&B disabled via YAML (wandb.enabled = false).")
        return False, None

    # read fields (may be empty strings)
    api_key = getattr(wandb_cfg, "api_key", "") or ""
    project = getattr(wandb_cfg, "project", "") or ""
    tags    = getattr(wandb_cfg, "tags", "")

    # optional API key: only set if non-empty
    if api_key:
        os.environ["WANDB_API_KEY"] = str(api_key)

    # build kwargs for wandb.init dynamically
    init_kwargs = {}

    if project != "":
        init_kwargs["project"] = project

    if tags != "":
        # allow string or list in YAML
        if isinstance(tags, str):
            tags = [tags]
        init_kwargs["tags"] = tags

    try:
        wandb.login()
        run = wandb.init(**init_kwargs)
        wandb.config.update(args)

        logging.info(f"W&B initialized with: {init_kwargs}")
        return True, run
    except Exception as e:
        logging.info(f"W&B initialization failed → disabling. Error: {e}")
        return False, None