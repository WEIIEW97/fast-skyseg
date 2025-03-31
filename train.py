import trainer

if __name__ == "__main__":
    skyseg_config = trainer.SkysegConfig()
    trainer = trainer.Trainer(skyseg_config)

    try:
        trainer.train()
    finally:
        trainer.close()
