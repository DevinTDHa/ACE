import os
import argparse

from os import path as osp

import torch

from tqdm import tqdm


# Diffusion Model imports
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

# core imports
from core.utils import generate_mask
from core.attacks_and_models import JointClassifierDDPM, get_attack

import matplotlib
import pdb
import sys

matplotlib.use("Agg")  # to disable display

from thesis_utils.counterfactuals import (
    CFResult,
    save_cf_results,
    update_results_oracle,
)
from thesis_utils.file_utils import (
    assert_paths_exist,
    create_result_dir,
    deterministic_run,
    dump_args,
    save_img_threaded,
)
from thesis_utils.image_folder_dataset import ImageFolderDataset, default_transforms
from thesis_utils.models import load_resnet


# =======================================================
# =======================================================
# Functions
# =======================================================
# =======================================================


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    print("Exception occurred:", exc_value)
    pdb.post_mortem(exc_traceback)


sys.excepthook = handle_exception


def create_args():
    defaults = dict(
        clip_denoised=True,  # Clipping noise
        batch_size=16,  # Batch size
        gpu="0",  # GPU index, should only be 1 gpu
        save_images=False,  # Saving all images
        num_samples=1,  # useful to sample few examples
        cudnn_deterministic=False,  # setting this to true will slow the computation time but will have identic results when using the checkpoint backwards
        # path args
        model_path="",  # DDPM weights path
        exp_name="exp",  # Experiment name (will store the results at Output/Results/exp_name)
        # attack args
        seed=0,  # Random seed
        attack_method="PGD",  # Attack method (currently 'PGD', 'C&W', 'GD' and 'None' supported)
        attack_iterations=50,  # Attack iterations updates
        attack_epsilon=255,  # L inf epsilon bound (will be devided by 255)
        attack_step=1.0,  # Attack update step (will be devided by 255)
        attack_joint=True,  # Set to false to generate adversarial attacks
        attack_joint_checkpoint=False,  # use checkpoint method for backward. Beware, this will substancially slow down the CE generation!
        attack_checkpoint_backward_steps=1,  # number of DDPM iterations per backward process. We highly recommend have a larger backward steps than batch size (e.g have 2 backward steps and batch size of 1 than 1 backward step and batch size 2)
        attack_joint_shortcut=False,  # Use DiME shortcut to transfer gradients. We do not recommend it.
        # dist args
        dist_l1=0.0,  # l1 scaling factor
        dist_l2=0.0,  # l2 scaling factor
        dist_schedule="none",  # schedule for the distance loss. We did not used any for our results
        # filtering args
        sampling_time_fraction=0.1,  # fraction of noise steps (e.g. 0.1 for 1000 smpling steps would be 100 out of 1000)
        sampling_stochastic=True,  # Set to False to remove the noise when sampling
        # post processing
        sampling_inpaint=0.15,  # Inpainting threshold
        sampling_dilation=15,  # Dilation size for the mask generation
        # query and target label
        # dataset
        image_size=256,  # Dataset image size
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    # Regression args
    parser.add_argument(
        "--rmodel_path",
        type=str,
        required=True,
        help="Path to the regression model.",
    )
    parser.add_argument(
        "--roracle_path",
        type=str,
        required=True,
        help="Path to the oracle model.",
    )

    def parse_target(value):
        if value == "-inf":
            return float("-inf")
        elif value == "inf":
            return float("inf")
        else:
            return float(value)

    parser.add_argument(
        "--target",
        type=parse_target,
        required=True,
        help="Target class for the attack. Can be a specific value or -inf or inf for untargeted attacks.",
    )
    parser.add_argument(
        "--stop_at",
        type=float,
        required=True,
        help="Target goal for the attack. Will stop the attack if the value is reached.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        required=False,
        help="Confidence to goal to stop the attack",
        default=0.05,
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help=(
            "Path to the samples folder. "
            "The folder should contain images and an optional data.csv file with two "
            "fields for filename and age."
            " If available, it will update true labels in the result dict."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        default="ace_results",
        help="Directory to save the results.",
    )

    args = parser.parse_args()
    return args


# =======================================================
# =======================================================
# Custom functions
# =======================================================
# =======================================================


@torch.no_grad()
def filter_fn(
    diffusion,
    attack,
    model,
    steps,
    x,
    stochastic,
    target,
    inpaint,
    dilation,
):

    indices = list(range(steps))[::-1]

    # 1. Generate pre-explanation
    with torch.enable_grad():
        pe, success, steps_done = attack.perturb(x, target)

    # 2. Inpainting: generates masks
    mask, dil_mask = generate_mask(x, pe, dilation)
    boolmask = (dil_mask < inpaint).float()

    ce = (pe.detach() - 0.5) / 0.5
    orig = (x.detach() - 0.5) / 0.5
    noise_fn = torch.randn_like if stochastic else torch.zeros_like

    for idx, t in enumerate(indices):

        # filter the with the diffusion model
        t = torch.tensor([t] * ce.size(0), device=ce.device)

        if idx == 0:
            ce = diffusion.q_sample(ce, t, noise=noise_fn(ce))
            noise_x = ce.clone().detach()

        if inpaint != 0:
            ce = ce * (1 - boolmask) + boolmask * diffusion.q_sample(
                orig, t, noise=noise_fn(ce)
            )

        out = diffusion.p_mean_variance(model, ce, t, clip_denoised=True)

        ce = out["mean"]

        if stochastic and (idx != (steps - 1)):
            noise = torch.randn_like(ce)
            ce += torch.exp(0.5 * out["log_variance"]) * noise

    ce = ce * (1 - boolmask) + boolmask * orig
    ce = (ce * 0.5) + 0.5
    ce = ce.clamp(0, 1)
    noise_x = ((noise_x * 0.5) + 0.5).clamp(0, 1)

    return ce, pe, noise_x, mask, success, steps_done


# =======================================================
# =======================================================
# Main
# =======================================================
# =======================================================


def load_model(args):
    model, respaced_diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    return model, respaced_diffusion


def get_data(args):
    def load_meta(meta_path: str):
        meta = {}
        with open(meta_path) as f:
            for line in f.readlines():
                filename, age = line.strip().split(",")
                meta[filename] = float(age) / 100

        return meta

    compose = default_transforms(args.size, ddpm=True)
    dataset = ImageFolderDataset(
        folder=args.image_folder, size=args.size, transform=compose
    )

    meta_path = os.path.join(args.image_folder, "data.csv")
    meta = None
    if os.path.exists(meta_path):
        meta = load_meta(meta_path)
    return dataset, meta


def main() -> None:
    args = create_args()

    # if args.merge_chunks:
    #     merge_all_chunks(args.chunks, args.output_path, args.exp_name)
    #     return

    respaced_steps = int(args.sampling_time_fraction * int(args.timestep_respacing))
    normal_steps = int(args.sampling_time_fraction * int(args.diffusion_steps))

    print("Using", respaced_steps, "respaced steps and", normal_steps, "normal steps")

    args.respaced_steps = respaced_steps
    args.normal_steps = normal_steps

    # ========================================
    # Setup the environment and results
    # ========================================

    deterministic_run(args.seed)
    assert_paths_exist(
        [args.model_path, args.rmodel_path, args.roracle_path, args.image_folder]
    )
    result_dir = create_result_dir(osp.join(args.output_path))
    dump_args(args, result_dir)

    # ========================================
    # load models
    # ========================================

    print("Loading Model and diffusion model")
    # respaced diffusion has the respaced strategy
    model, respaced_diffusion = load_model(args)

    print("Loading Regressor")
    classifier = load_resnet(args.rmodel_path)
    classifier.to(dist_util.dev()).eval()

    if args.attack_joint and not (
        args.attack_joint_checkpoint or args.attack_joint_shortcut
    ):
        joint_classifier = JointClassifierDDPM(
            classifier=classifier,
            ddpm=model,
            diffusion=respaced_diffusion,
            steps=respaced_steps,
            stochastic=args.sampling_stochastic,
        )
        joint_classifier.eval()

    # ========================================
    # load attack
    # ========================================

    def get_dist_fn():

        if args.dist_l2 != 0.0:
            l2_loss = (
                lambda x, x_adv: args.dist_l2
                * torch.linalg.norm((x - x_adv).view(x.size(0), -1), dim=1).sum()
            )
            any_loss = True

        if args.dist_l1 != 0.0:
            l1_loss = lambda x, x_adv: args.dist_l1 * (x - x_adv).abs().sum()
            any_loss = True

        if not any_loss:
            return None

        def dist_fn(x, x_adv):
            loss = 0
            if args.dist_l2 != 0.0:
                loss += l2_loss(x, x_adv)
            if args.dist_l1 != 0.0:
                loss += l1_loss(x, x_adv)
            return loss

        return dist_fn

    dist_fn = get_dist_fn()

    main_args = {
        "predict": (
            joint_classifier
            if args.attack_joint
            and not (args.attack_joint_checkpoint or args.attack_joint_shortcut)
            else classifier
        ),
        "loss_fn": "mse",  # we can implement here a custom loss fn
        "dist_fn": dist_fn,
        "eps": args.attack_epsilon / 255,
        "nb_iter": args.attack_iterations,
        "dist_schedule": args.dist_schedule,
        "binary": False,
        "step": args.attack_step / 255,
        "confidence_threshold": args.confidence_threshold,
    }

    attack = get_attack(
        args.attack_method,
        args.attack_joint and args.attack_joint_checkpoint,
        args.attack_joint and args.attack_joint_shortcut,
    )

    if args.attack_joint and (
        args.attack_joint_checkpoint or args.attack_joint_shortcut
    ):
        attack = attack(
            diffusion=respaced_diffusion,
            ddpm=model,
            steps=respaced_steps,
            stochastic=args.sampling_stochastic,
            backward_steps=args.attack_checkpoint_backward_steps,
            **main_args,
        )
    else:
        attack = attack(**main_args)  # Constructor

    dataset, meta = get_data(args)

    num_samples = (
        len(dataset)
        if args.num_samples is None
        else min(args.num_samples, len(dataset))
    )

    diffeocf_results: list[CFResult] = []

    # Other Ace Results
    pe_path = osp.join(result_dir, "pe")
    noise_path = osp.join(result_dir, "noise_x")
    mask_path = osp.join(result_dir, "mask")
    os.makedirs(pe_path, exist_ok=True)
    os.makedirs(noise_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)

    with tqdm(range(num_samples), desc="Running ACE") as pbar:
        for i in pbar:
            pbar.set_postfix_str(f"Processing: {f}")

            f, x = dataset[i]
            x = x.unsqueeze(0).to(dist_util.dev())
            x_reconstructed, y_initial = joint_classifier.initial(x)

            # sample image from the noisy_img
            # DHA: 1. Extract grads with JointClassifierDDPM.forward and perform PGD
            # DHA: 2. Create inpainting for final CE
            ce, pe, noise, pe_mask, success, steps_done = filter_fn(
                diffusion=respaced_diffusion,
                attack=attack,
                model=model,
                steps=respaced_steps,
                x=x.to(dist_util.dev()),
                stochastic=args.sampling_stochastic,
                target=args.target,
                inpaint=args.sampling_inpaint,
                dilation=args.sampling_dilation,
            )

            with torch.no_grad():
                y_final = joint_classifier.classifier(ce).item()
                cf_result = CFResult(
                    image_path=f,
                    x=x,
                    x_reconstructed=x_reconstructed,
                    x_prime=ce,
                    y_target=args.target,
                    y_initial_pred=y_initial,
                    y_final_pred=y_final,
                    success=success,
                    steps=steps_done,
                )
                if meta is not None:
                    y_true = meta[cf_result.image_name_base]
                    cf_result.update_y_true_initial(y_true)
                diffeocf_results.append(cf_result)

            save_img_threaded(pe, osp.join(pe_path, cf_result.image_name))
            save_img_threaded(noise, osp.join(noise_path, cf_result.image_name))
            save_img_threaded(pe_mask, osp.join(mask_path, cf_result.image_name))

            # Images for saving
            # noise = (noise * 255).to(dtype=torch.uint8).detach().cpu()
            # pe_mask = (pe_mask * 255).to(dtype=torch.uint8).detach().cpu()

    # Save the results for the diffeo_cf attacks
    del model
    del joint_classifier
    oracle = load_resnet(args.roracle_path).to("cuda")
    update_results_oracle(oracle, diffeocf_results, args.confidence_threshold)

    save_cf_results(diffeocf_results, args.result_dir)


if __name__ == "__main__":
    main()
