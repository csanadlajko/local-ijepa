from src.parser.parser import parse_jepa_args
from src.models.vit import VisionTransformer
from src.models.predictor import (
    ViTPredictor
)
from src.train.train_ijepa import (
    train, # main JEPA training loop for creating the representational space
    train_cls, # CLS token supervised training loop
    eval_cls, # cls evalutaion on the test dataset
    
)
from src.train.train_local_jepa import (
    train_local_jepa, # train JEPA model on COCO
    train_block_predictor, # finetune block predictor
    eval_block_predictor # evaluate finetuned JEPA model
)
from src.data_preprocess.dataloader import load_dataset
from src.utils.config_ijepa import get_model_config, init_weights, create_finetune_model
from src.utils.masking import Mask, CellMask
from src.utils.patch_metadata import PatchProcesser, BlockProcessor
from src.utils.logging_module import logger
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import datetime

args = parse_jepa_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Initializing models on device: {device}")

mask = Mask(device=device)
cell_mask = CellMask(device=device)

if __name__ == "__main__":
    jepa_loss_per_epoch = []
    accuracy_per_epoch = []
    cls_loss_per_epoch = []

    run_identifier: str = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%SZ")
    result_folder: str = args.result_folder

    datasets = load_dataset(args.dataset, logger, args.dataset_input, args.reverse_transform)

    if "error" in datasets:
        raise FileNotFoundError(datasets["error"])
    
    logger.info("Dataset loaded successfully!")

    train_loader, test_loader = datasets["train_loader"], datasets["test_loader"]

    teacher_model = VisionTransformer(
        img_size=args.image_size,
        patch_size=args.patch_size,
        in_chans=args.channels,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        mlp_dim=args.mlp_dim,
        drop_rate=args.teacher_dropout,
        num_classes=args.num_classes,
        logger=logger
    ).to(device)

    student_model = VisionTransformer(
        img_size=args.image_size,
        patch_size=args.patch_size,
        in_chans=args.channels,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        mlp_dim=args.mlp_dim,
        drop_rate=args.student_dropout,
        num_classes=args.num_classes,
        logger=logger
    ).to(device)

    tokenizer, text_encoder = None, None

    if args.multimodal_run == "y":
        logger.info("Loading pre-trained text tokenizer and encoder...")
        text_encoder = AutoModelForCausalLM.from_pretrained(args.sentence_encoder).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.sentence_encoder)
        logger.info("Tokenizer and text encoder loaded successfully!")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        logger.info("Skipping text encoder initialization as multimodal run is not set to true!")


    predictor = ViTPredictor(
        num_patches=teacher_model.patch_embed.num_patches,
        embed_dim=args.embed_dim,
        device=device,
        pred_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        num_classes=args.num_classes,
        num_targets=args.num_target,
        logger=logger
    ).to(device)

    patch_processor = PatchProcesser(
        patch_size=args.patch_size,
        logger=logger
    )

    block_proc = BlockProcessor(args.patch_size, args.image_size, logger=logger)

    block_predictor = create_finetune_model(
        geometry=args.geometry,
        jepa_embed_dim=args.embed_dim,
        proj_dim=args.finetune_embed_dim,
        tau=args.tau,
        r_max=args.r_max,
        margin=args.margin,
        class_parent=datasets["class_parent"]
    )

    if not block_predictor:
        raise ValueError("Block predictor couldn't be initialized. The project only supports the COCO dataset, and these embedding geometries: [euclidean, hyperbolic, spheric]")

    block_predictor.to(device)

    logger.info(f"Finetuning will take place in {args.geometry} geometry")

    # block_predictor = BlockTypePredictor(embed_dim=args.embed_dim, num_classes=args.num_block_categories, logger=logger).to(device)

    student_model.apply(init_weights)
    predictor.apply(init_weights)
    block_predictor.apply(init_weights)

    teacher_model.load_state_dict(student_model.state_dict())
    teacher_model.requires_grad_(False)

    model_config = get_model_config(
        student_model,
        predictor,
        args.lr,
        args.epochs,
        block_predictor,
        args.finetune_lr
    )

    student_scheduler = model_config["student_scheduler"]

    logger.info(
        f"total number of parameters approx.: {sum(p.numel() for p in student_model.parameters()) + sum(p.numel() for p in teacher_model.parameters()) + sum(p.numel() for p in predictor.parameters())}",
    )

    if args.multimodal_run == "y":
        logger.info(f"Starting training in MULTIMODAL mode for {args.epochs} epoch(s)!")
    else:
        logger.info(f"Starting training normal I-JEPA mode for {args.epochs} epoch(s)!")

    if args.skip_backbone == "n":
        for epoch in range(args.epochs):

            logger.info(f"=== EPOCH {epoch + 1}/{args.epochs} ===")

            if args.dataset != "pdl1" and args.dataset != "coco":
                ## train JEPA on regular classification tasks with cls token
                loss_epoch = train(
                    teacher_mod=teacher_model,
                    student_mod=student_model,
                    loader=train_loader,
                    optim_student=model_config["optim_student"],
                    optim_predictor=model_config["optim_predictor"],
                    predictor=predictor,
                    momentum=args.momentum,
                    ijepa_loss=model_config["ijepa_loss"],
                    device=device,
                    mask=mask,
                    multimodal=args.multimodal_run,
                    debug=args.debug
                )
            else:
                ## train JEPA on COCO with local representation classification
                loss_epoch = train_local_jepa(
                    teacher_mod=teacher_model,
                    student_mod=student_model,
                    loader=train_loader,
                    optim_student=model_config["optim_student"],
                    optim_predictor=model_config["optim_predictor"],
                    predictor=predictor,
                    momentum=args.momentum,
                    ijepa_loss=model_config["ijepa_loss"],
                    l_block_loss=args.l_block_loss,
                    device=device,
                    block_proc=block_proc,
                    normal_mask=mask
                )

            if (epoch+1) % 10 == 0:
                student_scheduler.step()
            jepa_loss_per_epoch.append(loss_epoch)

            if args.debug == "y" and (epoch+1) % 10 == 0:
                torch.save(student_model.state_dict(), f"{result_folder}/trained_student_jepa_{run_identifier}_{epoch+1}ep.pth")
                torch.save(teacher_model.state_dict(), f"{result_folder}/teacher_model_jepa_{run_identifier}_{epoch+1}ep.pth")
                torch.save(predictor.state_dict(), f"{result_folder}/trained_predictor_jepa_{run_identifier}_{epoch+1}ep.pth")

    else:
        logger.info(f"Skipping JEPA backbone training as skip_backbone parameter is set to {args.skip_backbone}")

    block_pred_loss = torch.nn.CrossEntropyLoss().to(device)

    if args.load_backbone == "y":
        match args.geometry.lower():
            case "euclidean":
                logger.info("loading pretrained euclidean models from results/euclidean")
                teacher_model.load_state_dict(torch.load("results/euclidean/teacher_model_jepa_2026-10-11T223242Z_20ep.pth", map_location=device))
                student_model.load_state_dict(torch.load("results/euclidean/trained_student_jepa_2026-10-11T223242Z_20ep.pth", map_location=device))
                predictor.load_state_dict(torch.load("results/euclidean/trained_predictor_jepa_2026-10-11T223242Z_20ep.pth", map_location=device))

            case "hyperbolic":
                logger.info("loading pretrained hyperbolic models from results/hyperbolic")
                teacher_model.load_state_dict(torch.load("results/hyperbolic/teacher_model_jepa_2026-10-11T223332Z_20ep.pth", map_location=device))
                student_model.load_state_dict(torch.load("results/hyperbolic/trained_student_jepa_2026-10-11T223332Z_20ep.pth", map_location=device))
                predictor.load_state_dict(torch.load("results/hyperbolic/trained_predictor_jepa_2026-10-11T223332Z_20ep.pth", map_location=device))

    for epoch in range(args.epochs):
        logger.info(f"=== Classification finetuning EPOCH {epoch+1}/{args.epochs} ===")
        if args.dataset != "pdl1" and args.dataset != "coco":
            cls_loss_at_epoch, accuracy_epoch = train_cls(
                student_model=student_model,
                train_dataset=train_loader,
                predictor=predictor,
                optim_cls=model_config["optim_cls"],
                cls_loss=model_config["cls_loss"],
                device=device,
                mask=mask,
                multimodal=False ## false in every case to prevent leakage !!
            )
        else:
            cls_loss_at_epoch, accuracy_epoch = train_block_predictor(
                student_mod=student_model,
                loader=train_loader,
                optim_block_predictor=model_config["optim_block_predictor"],
                predictor=predictor,
                device=device,
                block_processor=block_proc,
                loss_fn=block_pred_loss,
                normal_mask=mask,
                block_predictor=block_predictor,
                optim_student=model_config["optim_student"],
                optim_main_pred=model_config["optim_predictor"],
                l_cls=args.l_cls,
                l_parent=args.l_parent,
                l_depth=args.l_depth
            )
        accuracy_per_epoch.append(accuracy_epoch)
        cls_loss_per_epoch.append(cls_loss_at_epoch)

        if args.debug == "y" and (epoch+1) % 10 == 0:
            torch.save(student_model.state_dict(), f"{result_folder}/trained_student_{args.geometry}_{run_identifier}_{epoch+1}ep.pth")
            torch.save(teacher_model.state_dict(), f"{result_folder}/teacher_model_{args.geometry}_{run_identifier}_{epoch+1}ep.pth")
            torch.save(predictor.state_dict(), f"{result_folder}/trained_predictor_{args.geometry}_{run_identifier}_{epoch+1}ep.pth")
            if args.dataset == "coco":
                torch.save(block_predictor.state_dict(), f"{result_folder}/trained_block_pred_{args.geometry}_{run_identifier}_{epoch+1}ep.pth")

    if args.dataset != "pdl1" and args.dataset != "coco":
        cls_acc = eval_cls(
            model=student_model,
            test_dataset=test_loader,
            predictor=predictor,
            device=device,
            mask=mask,
            multimodal=False ## false in every case to prevent leakage !!
        )
    else:
        cls_acc = eval_block_predictor(
            student_model=student_model,
            predictor=predictor,
            test_loader=test_loader,
            device=device,
            normal_mask=mask,
            block_predictor=block_predictor,
            block_processor=block_proc
        )
    logger.info(f"Classification accuracy (map) is: {cls_acc}")
    # show_topk_accuracy(accuracy_per_epoch, run_identifier, result_folder)
    # show_loss_per_epoch(cls_loss_per_epoch, run_identifier, result_folder)
    # show_cls_data_per_epoch(accuracy_per_epoch, run_identifier, result_folder)