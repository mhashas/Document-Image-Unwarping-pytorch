from parser_options import ParserOptions
from util.general_functions import print_training_info
from constants import *
from core.trainers.trainer import Trainer

def main():
    args = ParserOptions().parse()  # get training options
    trainer = Trainer(args)

    print_training_info(args)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.run_epoch(epoch, split=TRAIN)

        if epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.run_epoch(epoch, split=TEST)

    trainer.run_epoch(trainer.args.epochs, split=VISUALIZATION)
    trainer.summary.writer.add_scalar('test/best_result', trainer.best_loss, args.epochs)
    trainer.summary.writer.close()
    trainer.save_network()


if __name__ == "__main__":
    main()