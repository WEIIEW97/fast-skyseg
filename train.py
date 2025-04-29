import trainer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="multi models config switch")
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--continue_training', action='store_true', default=False,
                       help='Continue training from checkpoint')
    parser.add_argument('--ckpt_path', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    model_type = args.model_type
    assert model_type in ["lraspp_mobilenet_v3_large", "fast_scnn", "bisenetv2", "u2net_full", "u2net_lite"]
    skyseg_config = trainer.SkysegConfig()
    if model_type == "lraspp_mobilenet_v3_large":
        skyseg_config.model = "lraspp_mobilenet_v3_large"
        skyseg_config.num_epochs = args.num_epochs
        if args.continue_training:
            skyseg_config.continue_training = True
        skyseg_config.ckpt_path = args.ckpt_path
    elif model_type == "fast_scnn":
        skyseg_config.model = "fast_scnn"
        skyseg_config.num_epochs = args.num_epochs
        if args.continue_training:
            skyseg_config.continue_training = True
        skyseg_config.ckpt_path = args.ckpt_path
    elif model_type == "bisenetv2":
        skyseg_config.model = "bisenetv2"
        skyseg_config.num_epochs = args.num_epochs
        if args.continue_training:
            skyseg_config.continue_training = True
        skyseg_config.ckpt_path = args.ckpt_path
    elif model_type == "u2net_full":
        skyseg_config.model = "u2net"
        skyseg_config.u2net_type = "full"
        skyseg_config.batch_size = 2
        skyseg_config.num_epochs = args.num_epochs
        if args.continue_training:
            skyseg_config.continue_training = True
        skyseg_config.ckpt_path = args.ckpt_path
    elif model_type == "u2net_lite":
        skyseg_config.model = "u2net"
        skyseg_config.u2net_type = "lite"
        skyseg_config.batch_size = 4
        skyseg_config.num_epochs = args.num_epochs
        if args.continue_training:
            skyseg_config.continue_training = True
        skyseg_config.ckpt_path = args.ckpt_path
    else:
        raise ValueError(f"unsupported model backbone type! : {model_type}")
    print(f"*"*50)
    print(f"you are running at backbone {model_type}")
    print(skyseg_config)
    
    trainer = trainer.Trainer(skyseg_config)

    try:
        trainer.train()
    finally:
        trainer.close()

